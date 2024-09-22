import os 
import torch
import torch.nn as nn
import numpy as np
import torch.utils
import random
from torch.utils.tensorboard import SummaryWriter
import json
import clip
import my_clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_eval
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from exit.utils import seed_worker, init_save_folder, set_seed
from torch.utils.data._utils.collate import default_collate
from models.clip_model import load_clip_model
import subprocess
from datetime import timedelta
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)

def load_len_estimator(args):
    model = LengthEstimator(512, 50)
    ckpt = torch.load(os.path.join('checkpoints', args.dataname, 'length_estimator', 'model', 'finest.tar'),
                      map_location=args.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model


def get_latest_commit_info():
    # Run the git command to get the latest commit hash and message
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=format:%H %s"],
        stdout=subprocess.PIPE,
        text=True,
        check=True
    )
    
    # Extract the output
    commit_info = result.stdout.strip().replace(' ', '_')
    return commit_info[:10]


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def text_embedding(clip_text):
    text = my_clip.tokenize(clip_text, truncate=True).to(args.device)
    feat_clip_text, word_emb = clip_model.encode_text(text)
    return feat_clip_text.float(), word_emb.float()


##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
args.exp_name = f'{get_latest_commit_info()}__{args.exp_name}'
args.out_dir = os.path.join(args.out_dir, 'eval', args.exp_name)

##### ---- DDP EVAL ---- #####
ddp_eval = args.ddp_eval
if ddp_eval:
    timeout_seconds = 3600  # Set timeout to 1 hour (3600 seconds)
    nccl_timeout = timedelta(seconds=timeout_seconds)

    world_size = args.world_size
    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    print(20*'---')
    print('ngpus_per_node: ', ngpus_per_node)
    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.init_method,
                            world_size=args.world_size,
                            rank=rank,
                            timeout=nccl_timeout)
    print("process group ready")
    print(f"From rank {rank} making model...")
    print(20*'---')
    master_process = rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    rank = 0
    local_rank = 0
    world_size = 1
    master_process = True
    device = args.device

args.rank = rank
args.local_rank = local_rank
args.world_size = world_size
args.device = device
args.master_process = master_process
args.batch_size = args.total_batch_size // world_size
args.val_ddp = False

os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.out_dir+'/html', exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir, args=args)
if master_process:
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    args.writer = writer
args.logger = logger

set_seed(args.seed)

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.val_or_test == 'val':
    is_test = False
    num_repeat = 1
    rand_pos = args.rand_pos
    repeat_time = 10
    if args.debug:
        num_repeat = 1
        rand_pos = args.rand_pos
        repeat_time = 2
elif args.val_or_test == 'test':
    is_test = True
    num_repeat = args.num_repeat_inner
    rand_pos = args.rand_pos
    repeat_time = 20

B_val = 32 if not args.debug else 1
val_dataset = dataset_TM_eval.T2MDataset(args.dataname, is_test, B_val, w_vectorizer, args=args)

if args.ddp_eval:
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size,
                                    rank=args.rank, shuffle=True, seed=args.seed)
else:
    val_sampler = None

def get_val_loader():
    g = torch.Generator()
    g.manual_seed(args.seed)
    return torch.utils.data.DataLoader(
                val_dataset,
                batch_size=32,
                shuffle=(val_sampler is None),
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
                pin_memory=args.pin_memory,
                worker_init_fn=seed_worker, generator=g)

val_loader = get_val_loader()

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, args.device)
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

print(20*'---')
print(f'Rank {rank}: len val_dataset: {len(val_dataset)}')
print(f'Rank {rank}: len val_loader: {len(val_loader)}')
print(20*'---')

##### ---- Network ---- #####
clip_model = load_clip_model(args)
clip_model.to(args.device)

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                    args.nb_code,
                    args.code_dim,
                    args.output_emb_width,
                    args.down_t,
                    args.stride_t,
                    args.width,
                    args.depth,
                    args.dilation_growth_rate)


trans_encoder = trans.Text2Motion_Transformer(net,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate, args=args)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
del ckpt
net.eval()
net.to(args.device)

if args.resume_trans is not None and not args.debug:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
    del ckpt
trans_encoder.train()
trans_encoder.to(args.device)

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
multi = []
repeat_time = repeat_time

set_seed(args.seed)
val_loader = get_val_loader()

from tqdm import tqdm
for i in tqdm(range(repeat_time)):
    
    if ddp_eval: val_sampler.set_epoch(i)

    best_fid, best_iter, best_div, best_top1, best_top2, \
    best_top3, best_matching, best_multi = eval_trans.evaluation_transformer(
        args.out_dir, val_loader, net, trans_encoder, 0, best_fid=1000, best_iter=0, 
        best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, 
        eval_wrapper=eval_wrapper, dataname=args.dataname, save = False, num_repeat=num_repeat, rand_pos=rand_pos,
        text_embedding=text_embedding, args=args)
    
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)
    multi.append(best_multi)


logger.info('final result:')
logger.info(f'fid: {sum(fid)/repeat_time}')
logger.info(f'div: {sum(div)/repeat_time}')
logger.info(f'top1: {sum(top1)/repeat_time}')
logger.info(f'top2: {sum(top2)/repeat_time}')
logger.info(f'top3: {sum(top3)/repeat_time}')
logger.info(f'matching: {sum(matching)/repeat_time}')
logger.info(f'multi: {sum(multi)/repeat_time}')

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
multi = np.array(multi)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)

if args.ddp_eval:
    destroy_process_group()