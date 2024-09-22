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
from exit.utils import init_save_folder, set_seed, seed_worker
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


def train_iter(
    net,
    optimizer,
    scheduler,
    stage="Train",
    total_iters=100,
    curr_iter=0,
    best_fid=0, best_iter=0, best_div=0, best_top1=0, best_top2=0, best_top3=0, best_matching=0,
):

    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
    nb_iter = curr_iter
    epoch = 0
    while True:
        if ddp: train_sampler.set_epoch(epoch)
        epoch += 1

        for batch in train_loader:
            nb_iter += 1

            if stage=='Warmup':
                optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
                if nb_iter % args.print_iter == 0:
                    if master_process: logger.info(f'current_lr: {current_lr}')

            gt_motion = batch.to(args.device).float()
            pred_motion, loss_commit, perplexity = net(gt_motion)
            # net(gt_motion, type='encode')

            loss_motion = Loss(pred_motion, gt_motion)
            loss_vel = Loss.forward_joint(pred_motion, gt_motion)
            loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if stage=='Train': scheduler.step()

            torch.cuda.synchronize() # wait for the GPU to finish work

            avg_recons += loss_motion.item()
            avg_perplexity += perplexity.item()
            avg_commit += loss_commit.item()

            if nb_iter % args.print_iter ==  0 and master_process:
                avg_recons /= args.print_iter
                avg_perplexity /= args.print_iter
                avg_commit /= args.print_iter

                if stage=='Train':
                    writer.add_scalar('./Train/L1', avg_recons, nb_iter)
                    writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
                    writer.add_scalar('./Train/Commit', avg_commit, nb_iter)

                logger.info(f"{stage}. Iter: {nb_iter} | Recons: {avg_recons:.5f} | Commit: {avg_commit:.5f} | PPL: {avg_perplexity:.2f} |")
                avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

            if master_process and (stage=='Train') and (nb_iter % args.eval_iter==0 or nb_iter==1):
                best_fid, best_iter, best_div, best_top1, best_top2, \
                    best_top3, best_matching = eval_trans.evaluation_vqvae(
                        args.out_dir, val_loader, net, nb_iter, best_fid,
                        best_iter, best_div, best_top1, best_top2,
                        best_top3, best_matching, eval_wrapper=eval_wrapper,
                        optimizer=optimizer, scheduler=scheduler, args=args)
        
            if nb_iter >= total_iters: break
        if nb_iter >= total_iters: break

    return net, optimizer, scheduler


##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
set_seed(args.seed)

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
if master_process:
    args.out_dir = os.path.join(args.out_dir, f'vq') # /{args.exp_name}
    # os.makedirs(args.out_dir, exist_ok = True)
    init_save_folder(args)
set_seed(args.seed)

##### ---- Logger ---- #####
if master_process:
    logger = utils_model.get_logger(args.out_dir, args.resume_pth, args=args)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    args.logger = logger
    args.writer = writer

########## ------------- Data -----------##############
w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit' : 
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
    
else :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

if master_process:
    logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Dataloader ---- #####
trainSet = dataset_VQ.VQMDataset(args.dataname,
                                    args.batch_size,
                                    window_size=args.window_size,
                                    num_workers=args.num_workers,
                                    unit_length=2**args.down_t,
                                    args=args)

if args.ddp:
    train_sampler = DistributedSampler(trainSet, num_replicas=args.world_size, rank=args.rank, shuffle=True)
else:
    train_sampler = None

train_loader = torch.utils.data.DataLoader(trainSet,
                                              args.batch_size,
                                              shuffle=(train_sampler is None),
                                              sampler=train_sampler,
                                              num_workers=args.num_workers,
                                              worker_init_fn=seed_worker,
                                              drop_last = True)


val_dataset = dataset_TM_eval.T2MDataset(args.dataname, False,
                                        32,
                                        w_vectorizer,
                                        unit_length=2**args.down_t,
                                        args=args)
val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=args.pin_memory)


if master_process:
    logger.info(f"len train dataset {len(trainSet)}")
    logger.info(f"len valid dataset {len(val_dataset)}")
    logger.info(f"len train loader  {len(train_loader)}")
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

net.train()
net.to(args.device)

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)
#### Move net to DDP ###################3
if ddp:
    net = DDP(net, device_ids=[local_rank])
raw_net = net.module if ddp else net # always contains the "raw" unwrapped net

if master_process:
    n = sum([p.numel() for k, p in raw_net.named_parameters()])
    logger.info(f"Number of transformer parameters: {n/1e6} M")

if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net.to(device)

##### ---- Load model or Resume ---- #####
best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching = 1000, 0, 100, 0, 0, 0, 100

if args.resume_pth is None and master_process:
    best_fid, best_iter, best_div, best_top1,\
        best_top2, best_top3, best_matching = eval_trans.evaluation_vqvae(
            args.out_dir, val_loader, raw_net, 0, best_fid=1000, best_iter=0, best_div=100,
            best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper,
            optimizer=optimizer, scheduler=scheduler, draw=False, args=args, save=False)

train_iter(
    net,
    optimizer,
    scheduler,
    "Warmup",
    args.warm_up_iter,
    0,
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching)

train_iter(
    net,
    optimizer,
    scheduler,
    "Train",
    args.total_iters,
    0,
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching)