import os 
import torch
import numpy as np
import time as ti
import random

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip
import my_clip
import math

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from models.clip_model import load_clip_model
import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset.dataset_tokenize import save_tokens
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from exit.utils import load_last_transformer, load_trans_from_MMM, load_last_opt_sch, show, set_seed, seed_worker
from exit.utils import load_vq_pretrained, generate_src_mask, init_save_folder, uniform, cosine_schedule, save_random_state, restore_random_state
from einops import rearrange
import torch.nn.functional as F
from exit.utils import base_dir
import inspect
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate
from datetime import timedelta
import subprocess


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


def create_causal_mask(seq_len, bs):
    """
    Create a causal mask for autoregressive training.
    
    :param seq_len: Length of the sequence
    :return: A (seq_len, seq_len) mask tensor with 0s in the upper triangular part and 1s in the lower triangular part.
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).unsqueeze(0).repeat(bs, 1, 1)
    return mask


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def random_value(p=0.01, value_1=[0, 0.5], value_2=[0.5, 1]):
    # Generate a random number between 0 and 1
    random_num = torch.rand(1).item()
    # Return value_1 if random number is less than the prob, otherwise return value_2
    return value_1 if random_num < p else value_2

def text_embedding(clip_text):
    text = my_clip.tokenize(clip_text, truncate=True).to(args.device)
    feat_clip_text, word_emb = clip_model.encode_text(text)
    return feat_clip_text.float(), word_emb.float()

    # text_max_len = (text!=0).sum(dim=1)
    # text_mask = generate_src_mask(77, text_max_len).to(device)
    # text_mask = text_mask[:, :max(text_max_len)]
    # word_emb = word_emb[:, :max(text_max_len)]

    # return feat_clip_text, mid_features, eot_features, word_emb, text_mask


def get_loss_acc_masking(cls_pred, seq_mask_loss, target, *argss, **kwargs):
    weights = seq_mask_loss / (seq_mask_loss.sum(-1).unsqueeze(-1) * seq_mask_loss.shape[0])
    cls_pred_seq_masked = cls_pred[seq_mask_loss, :].view(-1, cls_pred.shape[-1])
    target_seq_masked = target[seq_mask_loss]
    weight_seq_masked = weights[seq_mask_loss]
    loss = F.cross_entropy(cls_pred_seq_masked, target_seq_masked, reduction = 'none')
    loss = (loss * weight_seq_masked).sum()

    probs_seq_masked = torch.softmax(cls_pred_seq_masked, dim=-1)
    _, cls_pred_seq_masked_index = torch.max(probs_seq_masked, dim=-1)
    target_seq_masked = torch.masked_select(target, seq_mask_loss)
    right_seq_masked = (cls_pred_seq_masked_index == target_seq_masked).sum()
    acc = right_seq_masked*100/seq_mask_loss.sum()

    return loss, acc


def get_acc(cls_pred, target, mask):
    cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
    target_all = torch.masked_select(target, mask)
    probs = torch.softmax(cls_pred, dim=-1)
    _, cls_pred_index = torch.max(probs, dim=-1)
    right_num = (cls_pred_index == target_all).sum()
    return right_num*100/mask.sum()

##### ---- Args ---- #####
args = option_trans.get_args_parser()
args.exp_name = f'{get_latest_commit_info()}__{args.exp_name}'
args.out_dir = os.path.join(args.out_dir, args.exp_name)

##### ---- DDP ---- #####
ddp = args.ddp
if ddp:
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

########## ------------- Seed -----------##############
set_seed(args.seed + args.rank)

########## ------------- DIRS -----------##############
# [TODO] make the 'output/' folder as arg
args.codebook_dir = pjoin(os.sep.join(os.path.normpath(args.vq_pretrained_path).split(os.sep)[:-1]), 'codebook')
os.makedirs(args.codebook_dir, exist_ok = True)

os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.out_dir+'/html', exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir, args=args)
if master_process:
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    args.writer = writer
args.logger = logger

##### ---- Networks ---- #####
##### ----   CLIP   ---- #####
clip_model = load_clip_model(args)
clip_model.to(local_rank)

##### ----   VQVAE   ---- #####
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)

net = load_vq_pretrained(net, args)
net.to(local_rank)

if master_process:
    if len(os.listdir(args.codebook_dir)) == 0:
        save_tokens(args, net)
    logger.info(f"Number of Codes: {len(os.listdir(args.codebook_dir))}")

### ---- IDs ---- ####
MASK_ID = net.mask_id
PAD_ID  = net.pad_id
END_ID  = net.end_id

##### ----   Transformer   ---- #####
model = trans.Text2Motion_Transformer(vqvae=net,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate,
                                args=args)

curr_epoch = 0
if args.resume_pth is not None:
    model, curr_epoch = load_last_transformer(model, args)
else:
    if master_process: args.logger.info("Train from scractch.")

model.to(local_rank)

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, 
                                      args.weight_decay, model, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=args.lr_scheduler, 
                                                 gamma=args.gamma)
loss_ce = torch.nn.CrossEntropyLoss(reduction='none')

if args.resume_pth is not None:
    optimizer, scheduler = load_last_opt_sch(optimizer, scheduler, args)

if ddp:
    model = DDP(model, device_ids=[local_rank], 
                find_unused_parameters=args.find_unused_parameters)
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

if master_process:
    n = sum([p.numel() for k, p in raw_model.named_parameters() if 'vqvae' not in k])
    logger.info(f"Number of transformer parameters: {n/1e6} M")

############ eval_wrapper ##############
dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' \
    if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
wrapper_opt = get_opt(dataset_opt_path, torch.device(args.device))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Dataloader ---- #####
from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
train_dataset = dataset_TM_train.T2MDataset(
    args.dataname, args.batch_size, args.nb_code, args.codebook_dir,
    unit_length=2**args.down_t, args=args)
# if master_process:
val_dataset = dataset_TM_eval.T2MDataset(
    args.dataname, False, 32, w_vectorizer,
    num_workers=args.num_workers, args=args)

train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size,
                                    rank=args.rank, shuffle=True, seed=args.seed) if ddp else None

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),  # No shuffle if using DistributedSampler
    num_workers=args.num_workers,
    drop_last=True,
    pin_memory=args.pin_memory)

def get_val_loader():
    g = torch.Generator()
    g.manual_seed(args.seed)
    return torch.utils.data.DataLoader(
                val_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
                pin_memory=args.pin_memory,
                worker_init_fn=seed_worker, generator=g)

val_loader = get_val_loader()

print(20*'---')
print(f'Rank {rank}: len train_dataset: {len(train_dataset)}')
print(f'Rank {rank}: len train_loader: {len(train_loader)}')
print(20*'---')

if master_process:
    logger.info(f"Number of train samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")
    logger.info(f'len train_loader: {len(train_loader)}')
    logger.info(f'len val_loader: {len(val_loader)}')


##### ---- Training ---- #####
best_fid_ever, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching = 1000, 1000, 0, 100, 0, 0, 0, 100
repeat_time = 10 if not args.debug else 1

running_loss = 0
running_acc = 0

print(20*'---')
print("Start Training ...", rank)
print(20*'---')

iter = curr_epoch * len(train_loader)
epoch = curr_epoch
model.train()

while True:
    if ddp: train_sampler.set_epoch(epoch)
    epoch += 1
    for i, batch in enumerate(train_loader):
        iter += 1
        clip_text, m_tokens, m_tokens_len = batch
        m_tokens, m_tokens_len = m_tokens.to(local_rank), m_tokens_len.to(local_rank)

        target = m_tokens
        batch_size, max_len = target.shape[:2]
        feat_clip_text, word_emb  = text_embedding(clip_text)

        input_indices, seq_mask, seq_mask_no_end, seq_mask_with_end, mask_token, *_ = \
            raw_model.permuted_corruption(target, max_len, m_tokens_len, args=args)

        cls_pred = model(input_indices, feat_clip_text, src_mask=seq_mask, word_emb=word_emb)

        loss, acc = get_loss_acc_masking(cls_pred, seq_mask_no_end, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize() # wait for the GPU to finish work

        if iter % args.print_iter == 0 and master_process:
            average_loss = loss.item() # log loss and acc just for the master node
            average_acc  = acc.item()

            msg = f"| Epoch: {epoch} | Iter: {iter:3d} | Loss: {average_loss:.5f} | ACC: {average_acc:.4f} |"
            logger.info(msg)

            writer.add_scalar('./Loss/average', average_loss, iter)
            writer.add_scalar('./ACC/average', average_acc, iter)
            
            # [INFO] log mask/nomask separately
            no_mask_token = ~mask_token * seq_mask_no_end
            writer.add_scalar('./ACC/masked', get_acc(cls_pred, target, mask_token), iter)
            writer.add_scalar('./ACC/no_masked', get_acc(cls_pred, target, no_mask_token), iter)
    
        if (iter % args.eval_iter ==  0) or (iter == args.total_iters) or (iter==1):
            random_state = save_random_state()
            set_seed(args.seed)
            val_loader = get_val_loader()
            ##########################################################
            if rank == 0 or not ddp: # Confidence-Based Sampling
                raw_model.sample = raw_model.Confidence_Based_Sampling
                raw_model.args.start_ids_in_sampling = "RANDOM"
                
                best_fid_list = []
                for i_fid in range(repeat_time):
                    curr_fid, best_iter, best_div, best_top1, \
                        best_top2, best_top3, best_matching, best_multi = eval_trans.evaluation_transformer(
                                    args.out_dir, val_loader, net, raw_model, iter, best_fid,
                                    best_iter, best_div, best_top1, best_top2, best_top3, 
                                    best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, 
                                    dataname=args.dataname, num_repeat=1, text_embedding=text_embedding,
                                    args=args, optimizer=optimizer, 
                                    scheduler=scheduler, epoch=epoch, save=i_fid==0)
                    best_fid_list.append(curr_fid)
                fid = sum(best_fid_list) / len(best_fid_list)
                if fid < best_fid_ever:
                    args.logger.info(20*'-----')
                    msg = f"--> --> \t FID CONFIDENCE Improved from {best_fid_ever:.5f} to {fid:.5f} !!!"
                    args.logger.info(msg)
                    best_fid_ever = fid
                    torch.save({'trans' : raw_model.state_dict(), 'iter': iter},
                               os.path.join(args.out_dir, f'CBS_net_best_fid.pth'))
            
            ##########################################################
            if rank == 1 or not ddp: # Order-Agnostic Autoregressive Sampling
                raw_model.sample = raw_model.Order_Agnostic_Autoregressive_Sampling
                raw_model.args.start_ids_in_sampling = "RANDOM"
                
                best_fid_list = []
                for i_fid in range(repeat_time):
                    curr_fid, best_iter, best_div, best_top1, \
                        best_top2, best_top3, best_matching, best_multi = eval_trans.evaluation_transformer(
                                    args.out_dir, val_loader, net, raw_model, iter, best_fid,
                                    best_iter, best_div, best_top1, best_top2, best_top3, 
                                    best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, 
                                    dataname=args.dataname, num_repeat=1, text_embedding=text_embedding,
                                    args=args, optimizer=optimizer, 
                                    scheduler=scheduler, epoch=epoch, save=False)
                    best_fid_list.append(curr_fid)
                fid = sum(best_fid_list) / len(best_fid_list)
                if fid < best_fid_ever:
                    args.logger.info(20*'-----')
                    msg = f"--> --> \t FID OAAS Improved from {best_fid_ever:.5f} to {fid:.5f} !!!"
                    args.logger.info(msg)
                    best_fid_ever = fid
                    torch.save({'trans' : raw_model.state_dict(), 'iter': iter},
                               os.path.join(args.out_dir, f'OAAS_net_best_fid.pth'))

            restore_random_state(random_state)
            model.train()
            net.eval()

    if iter >= args.total_iters: break

# Cleanup
if args.ddp:
    destroy_process_group()

