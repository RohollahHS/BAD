import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## dataloader
    
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--total_batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq_len', type=int, default=64, help='training motion length')
    
    ## optimization
    parser.add_argument('--total_iters', default=350_000, type=int)
    parser.add_argument('--warm_up_iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr_scheduler', default=[150000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay_option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    ## vqvae arch
    parser.add_argument("--code_dim", type=int, default=32, help="embedding dimension")
    parser.add_argument("--nb_code", type=int, default=8192, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq_act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')

    ## gpt arch
    parser.add_argument("--block_size", type=int, default=51, help="seq len")
    parser.add_argument("--embed_dim_gpt", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--clip_dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num_layers", type=int, default=9, help="nb of transformer layers")
    parser.add_argument("--num_local_layer", type=int, default=2, help="nb of transformer local layers")
    parser.add_argument("--n_head_gpt", type=int, default=16, help="nb of heads")
    parser.add_argument("--ff_rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop_out_rate", type=float, default=0.1, help="dropout ratio in the pos encoding")
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## resume
    parser.add_argument("--resume_pth", type=str, help='resume vq pth')
    parser.add_argument("--resume_trans", type=str, default=None, help='resume gpt pth')
    
    
    ## output directory 
    parser.add_argument('--out_dir', type=str, default='output', help='output directory')
    parser.add_argument('--exp_name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out_dir')
    parser.add_argument('--vq_name', type=str, default='VQVAE', help='name of the generated dataset .npy, will create a file inside out_dir')
    ## other
    parser.add_argument('--print_iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval_iter', default=5000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    parser.add_argument('--eval_seed', default=1234, type=int, help='seed for initializing validation.', choices=[1234])
    parser.add_argument("--if_maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=.5, help='keep rate for gpt training')
    

##########################################################################################################################################
    ## new options
    parser.add_argument('--sampling_type', default="CBS", choices=["CBS", "OAAS", "RS"], type=str, 
                        help='CBS: Confidence_Based_Sampling - OAAS: Order_Agnostic_Autoregressive_Sampling - RS: Random_Sampling')
    parser.add_argument('--start_ids_in_sampling', choices=['ARANGED', 'RANDOM'], default='RANDOM', help='if RANDOM, the generation starts with a random ordering. If ARANGED, the generation starts always with [1, 2, 3, .., T]')
    parser.add_argument('--unmasked_tokens_not_attend_to_mask_tokens', action='store_true', help='prohibits unmasked tokens from attending to mask tokens')
    parser.add_argument('--use_relative_position', action='store_true')
    parser.add_argument('--z_0_attend_to_all', action='store_true', help='Specifies the causality condition for mask tokens, where each mask token attends to the last T-p+1 mask tokens. If z_0_attend_to_all is not activated, each mask token attends to the first p mask tokens')
    parser.add_argument('--mask_scheduler', default='cosine', type=str, choices=['cosine', 'linear', 'square'])
    parser.add_argument('--max_steps_generation', default=10, type=int, help='The number of iterations during generation')

##########################################################################################################################################

    ## Transformer options
    parser.add_argument('--time_cond', action='store_true')
    parser.add_argument('--max_length', default=50, type=int)

    ## Paths
    parser.add_argument("--vq_pretrained_path", type=str, help='resume vq pth', default='checkpoints/pretrained_MMM/vq/2024-06-03-20-22-07_retrain/net_last.pth')

    ### Training Options
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', action='store_false')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test_time', action='store_true')
    parser.add_argument('--maxdata', type=int, default=128, help="Max number of data for debug")
    parser.add_argument('--val_or_test', type=str, default='test', choices=['test', 'val'])


    ### distributed
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--ddp', action='store_true', help='')
    parser.add_argument('--ddp_eval', action='store_true', help='')
    parser.add_argument('--find_unused_parameters', action='store_true')
    parser.add_argument('--val_ddp', action='store_true')

    ### Eval Options
    parser.add_argument('--num_repeat_inner', default=11, type=int)
    parser.add_argument('--top_k', default=0, type=int, help="if 1, argmax")
    parser.add_argument('--top_p', default=0.0, type=float, help="between 0 and 1")
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--rand_pos', action='store_true')
    parser.add_argument('--edit_task', type=str, choices=['inbetween', 'outpainting', 'prefix', 'suffix', 'upperbody'], default='inbetween')
    parser.add_argument('--use_length_estimator', action='store_true')

    ## Generate Options
    parser.add_argument('--caption', type=str, help='text', default='a person jauntily skips forward.')
    parser.add_argument('--length', type=int, help='length', default=-1)
    parser.add_argument('--repeat_times_generation', type=int, help='length', default=1)

    parser.add_argument('--temporal_editing', action='store_true', help='if you want to do temporal editing, you have to activate this.')
    parser.add_argument('--caption_inbetween', default='a man walks in a clockwise circle an then sits', help='the caption for 4 temporal editing tasks.')


    return parser.parse_args()
