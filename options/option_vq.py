import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory', choices=['kit', 't2m', 'both'])
    parser.add_argument('--total_batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--window_size', type=int, default=64, help='training motion length')

    ## optimization
    parser.add_argument('--total_iters', default=300_000, type=int)
    parser.add_argument('--warm_up_iter', default=1000, type=int)
    parser.add_argument('--reset_codebook_every', default=1000, type=int)
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr_scheduler', default=[200000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper_parameter for the commitment loss")
    parser.add_argument("--orth", type=float, default=0.01)
    parser.add_argument('--loss_vel', type=float, default=0.5, help='hyper_parameter for the velocity loss')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')
    
    ## vqvae arch
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq_act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq_norm', type=str, default=None, help='dataset directory')
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume_pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume_gpt", type=str, default=None, help='resume pth for GPT')
    
    
    ## output directory 
    parser.add_argument('--out_dir', type=str, default='output', help='output directory')
    parser.add_argument('--results_dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual_name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp_name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out_dir')
    ## other
    parser.add_argument('--print_iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval_iter', default=5000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis_gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb_vis', default=20, type=int, help='nb of visualizations')
    
    parser.add_argument('--sep_uplow', action='store_true', help='whether visualize GT motions')

    ### New Options
    parser.add_argument('--pin_memory', action='store_false')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--maxdata', default=124, type=int)

    ### VQVAE
    parser.add_argument("--code_dim", type=int, default=32, help="embedding dimension")
    parser.add_argument("--nb_code", type=int, default=8192, help="nb of embedding")

    ### distributed
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--ddp', action='store_true', help='')

    return parser.parse_args()