import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
from tqdm import tqdm
from exit.utils import load_vqvae_from_MMM, init_save_folder, load_last_vqvae, set_seed, seed_worker
from models.vqvae_sep import VQVAE_SEP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)


##### ---- DDP ---- #####
ddp = args.ddp
if ddp:
    world_size = args.world_size
    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    print(20*'-----')
    print('ngpus_per_node: ', ngpus_per_node)
    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    print("process group ready")
    print(f"From rank {rank} making model...")
    print(20*'-----')
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

########## ------------- Seed -----------##############
set_seed(args.seed)
########## ------------- DIRS -----------##############
if master_process:
    args.out_dir = os.path.join(args.out_dir, f'vq', 'eval') # /{args.exp_name}
    # os.makedirs(args.out_dir, exist_ok = True)
    init_save_folder(args)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir, args=args)
writer = SummaryWriter(args.out_dir)
if master_process:
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
args.logger = logger
args.writer = writer

w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit' : 
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
else:
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, device)
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

val_dataset = dataset_TM_eval.T2MDataset(args.dataname, True,
                                        32,
                                        w_vectorizer,
                                        unit_length=2**args.down_t,
                                        args=args)

val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=args.pin_memory)

if master_process:
    logger.info(f"len valid dataset {len(val_dataset)}")
    logger.info(f"len valid loader  {len(val_loader)}")

##### ---- Network ---- #####
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                    args.nb_code,
                    args.code_dim,
                    args.output_emb_width,
                    args.down_t,
                    args.stride_t,
                    args.width,
                    args.depth,
                    args.dilation_growth_rate,
                    args.vq_act,
                    args.vq_norm)
net.eval()
net.to(local_rank)

if master_process:
    n = sum([p.numel() for k, p in net.named_parameters()])
    logger.info(f"Number of transformer parameters: {n/1e6} M")

if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')

    try:
        net.load_state_dict(ckpt['net'], strict=True)
        del ckpt
    except:
        sd = {}
        for k, v in ckpt['net'].items():
            new_k = k.split('module.')[-1]
            sd[k] = v
        net.load_state_dict(sd, strict=True)
        del sd
        del ckpt

##### ------ warm-up ------- #####
fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
repeat_time = 10

for i in range(repeat_time):
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching \
        = eval_trans.evaluation_vqvae(
            args.out_dir, val_loader, net, 0, best_fid=1000, best_iter=0,\
            best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100,
            eval_wrapper=eval_wrapper, args=args)

    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)

logger.info('final result:')
logger.info(f'fid: {sum(fid)/repeat_time}')
logger.info(f'div: {sum(div)/repeat_time}')
logger.info(f'top1: {sum(top1)/repeat_time}')
logger.info(f'top2: {sum(top2)/repeat_time}')
logger.info(f'top3: {sum(top3)/repeat_time}')
logger.info(f'matching: {sum(matching)/repeat_time}')

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)
