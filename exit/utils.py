def get_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model

import numpy as np
import torch
import random
from utils.motion_process import recover_from_ric
import copy
import plotly.graph_objects as go
import shutil
import datetime
import os
import math
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap


t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
kit_kinematic_chain = [0]

kit_bone = [[0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [0, 16], [16, 17], [17, 18], [18, 19], [19, 20], [0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [6, 7], [3, 8], [8, 9], [9, 10]]
t2m_bone = [[0,2], [2,5],[5,8],[8,11],
            [0,1],[1,4],[4,7],[7,10],
            [0,3],[3,6],[6,9],[9,12],[12,15],
            [9,14],[14,17],[17,19],[19,21],
            [9,13],[13,16],[16,18],[18,20]]
kit_kit_bone = kit_bone + (np.array(kit_bone)+21).tolist()
t2m_t2m_bone = t2m_bone + (np.array(t2m_bone)+22).tolist()

def axis_standard(skeleton):
    skeleton = skeleton.copy()
#     skeleton = -skeleton
    # skeleton[:, :, 0] *= -1
    # xyz => zxy
    skeleton[..., [1, 2]] = skeleton[..., [2, 1]]
    skeleton[..., [0, 1]] = skeleton[..., [1, 0]]
    return skeleton

def visualize_2motions(motion1, std, mean, dataset_name, length, motion2=None, save_path=None, args=None):
    motion1 = motion1 * std + mean
    if motion2 is not None:
        motion2 = motion2 * std + mean
    if dataset_name == 'kit':
        first_total_standard = 60
        bone_link = kit_bone
        if motion2 is not None:
            bone_link = kit_kit_bone
        joints_num = 21
        scale = 1/1000
    else:
        first_total_standard = 63
        bone_link = t2m_bone
        if motion2 is not None:
            bone_link = t2m_t2m_bone
        joints_num = 22
        scale = 1#/1000
    joint1 = recover_from_ric(torch.from_numpy(motion1).float(), joints_num).numpy()
    if motion2 is not None:
        joint2 = recover_from_ric(torch.from_numpy(motion2).float(), joints_num).numpy()
        joint_original_forward = np.concatenate((joint1, joint2), axis=1)
    else:
        joint_original_forward = joint1
    animate3d(joint_original_forward[:length]*scale, 
              BONE_LINK=bone_link, 
              first_total_standard=first_total_standard, 
              save_path=save_path+'.html') # 'init.html'
    
    all_motions = torch.from_numpy(joint1).permute(1, 2, 0).unsqueeze(0).numpy()
    all_text = [args.text]
    all_lengths = np.array([length])

    out_path = save_path
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': 1, 'num_repetitions': 1})

    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))



    print(f"saving visualizations to [{out_path}]...")
    skeleton = t2m_kinematic_chain if dataset_name=='t2m' else kit_kinematic_chain

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(False)

    for sample_i in range(1):
        rep_files = []
        for rep_i in range(1):
            caption = all_text[rep_i*1 + sample_i]
            length = all_lengths[rep_i*1 + sample_i]
            motion = all_motions[rep_i*1 + sample_i].transpose(2, 0, 1)[:length]
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=dataset_name, title=caption, fps=20)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(out_path,
                                               row_print_template, all_print_template, row_file_template, all_file_template,
                                               caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

def save_multiple_samples(out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    num_repetitions = 1
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={num_repetitions}' if num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == 1:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template

def animate3d(skeleton, BONE_LINK=t2m_bone, first_total_standard=-1, root_path=None, root_path2=None, save_path=None, axis_standard=axis_standard, axis_visible=True):
    # [animation] https://community.plotly.com/t/3d-scatter-animation/46368/6
    
    SHIFT_SCALE = 0
    START_FRAME = 0
    NUM_FRAMES = skeleton.shape[0]
    skeleton = skeleton[START_FRAME:NUM_FRAMES+START_FRAME]
    skeleton = axis_standard(skeleton)
    if BONE_LINK is not None:
        # ground truth
        bone_ids = np.array(BONE_LINK)
        _from = skeleton[:, bone_ids[:, 0]]
        _to = skeleton[:, bone_ids[:, 1]]
        # [f 3(from,to,none) d]
        bones = np.empty(
            (_from.shape[0], 3*_from.shape[1], 3), dtype=_from.dtype)
        bones[:, 0::3] = _from
        bones[:, 1::3] = _to
        bones[:, 2::3] = np.full_like(_to, None)
        display_points = bones
        mode = 'lines+markers'
    else:
        display_points = skeleton
        mode = 'markers'
    # follow this thread: https://community.plotly.com/t/3d-scatter-animation/46368/6
    fig = go.Figure(
        data=go.Scatter3d(  x=display_points[0, :first_total_standard, 0], 
                            y=display_points[0, :first_total_standard, 1],
                            z=display_points[0, :first_total_standard, 2], 
                            name='Nodes0',
                            mode=mode, 
                            marker=dict(size=3, color='blue',)), 
                            layout=go.Layout(
                                scene=dict(aspectmode='data', 
                                camera=dict(eye=dict(x=3, y=0, z=0.1)))
                                )
                            )
    if first_total_standard != -1:
        fig.add_traces(data=go.Scatter3d(  
                                x=display_points[0, first_total_standard:, 0], 
                                y=display_points[0, first_total_standard:, 1],
                                z=display_points[0, first_total_standard:, 2], 
                                name='Nodes1',
                                mode=mode, 
                                marker=dict(size=3, color='red',)))

    if root_path is not None:
        root_path = axis_standard(root_path)
        fig.add_traces(data=go.Scatter3d(  
                                    x=root_path[:, 0], 
                                    y=root_path[:, 1],
                                    z=root_path[:, 2], 
                                    name='root_path',
                                    mode=mode, 
                                    marker=dict(size=2, color='green',)))
    if root_path2 is not None:
        root_path2 = axis_standard(root_path2)
        fig.add_traces(data=go.Scatter3d(  
                                    x=root_path2[:, 0], 
                                    y=root_path2[:, 1],
                                    z=root_path2[:, 2], 
                                    name='root_path2',
                                    mode=mode, 
                                    marker=dict(size=2, color='red',)))

    frames = []
    # frames.append({'data':copy.deepcopy(fig['data']),'name':f'frame{0}'})

    def update_trace(k):
        fig.update_traces(x=display_points[k, :first_total_standard, 0],
            y=display_points[k, :first_total_standard, 1],
            z=display_points[k, :first_total_standard, 2],
            mode=mode,
            marker=dict(size=3, ),
            # traces=[0],
            selector = ({'name':'Nodes0'}))
        if first_total_standard != -1:
            fig.update_traces(x=display_points[k, first_total_standard:, 0],
                y=display_points[k, first_total_standard:, 1],
                z=display_points[k, first_total_standard:, 2],
                mode=mode,
                marker=dict(size=3, ),
                # traces=[0],
                selector = ({'name':'Nodes1'}))

    for k in range(0, len(display_points)):
        update_trace(k)
        frames.append({'data':copy.deepcopy(fig['data']),'name':f'frame{k}'})
    update_trace(0)

    # frames = [go.Frame(data=[go.Scatter3d(
    #     x=display_points[k, :, 0],
    #     y=display_points[k, :, 1],
    #     z=display_points[k, :, 2],
    #     mode=mode,
    #     marker=dict(size=3, ))],
    #     traces=[0],
    #     name=f'frame{k}'
    # )for k in range(len(display_points))]
    
    
    
    fig.update(frames=frames)

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {"pad": {"b": 10, "t": 60},
         "len": 0.9,
         "x": 0.1,
         "y": 0,

         "steps": [
            {"args": [[f.name], frame_args(0)],
             "label": str(k),
             "method": "animate",
             } for k, f in enumerate(fig.frames)
        ]
        }
    ]

    fig.update_layout(
        updatemenus=[{"buttons": [
            {
                "args": [None, frame_args(1000/25)],
                "label": "Play",
                "method": "animate",
            },
            {
                "args": [[None], frame_args(0)],
                "label": "Pause",
                "method": "animate",
            }],

            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": 0.1,
            "y": 0,
        }
        ],
        sliders=sliders
    )
    range_x, aspect_x = get_range(skeleton, 0)
    range_y, aspect_y = get_range(skeleton, 1)
    range_z, aspect_z = get_range(skeleton, 2)

    fig.update_layout(scene=dict(xaxis=dict(range=range_x, visible=axis_visible),
                                 yaxis=dict(range=range_y, visible=axis_visible),
                                 zaxis=dict(range=range_z, visible=axis_visible)
                                 ),
                      scene_aspectmode='manual',
                      scene_aspectratio=dict(
                          x=aspect_x, y=aspect_y, z=aspect_z)
                      )

    fig.update_layout(sliders=sliders)
    fig.show()
    if save_path is not None:
        fig.write_html(save_path, auto_open=False, include_plotlyjs='cdn', full_html=False)


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 't2m':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.clear()
        # ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()



def get_range(skeleton, index):
    _min, _max = skeleton[:, :, index].min(), skeleton[:, :, index].max()
    return [_min, _max], _max-_min

# [INFO] from http://juditacs.github.io/2018/12/27/masked-attention.html
def generate_src_mask(T, length):
    B = len(length)
    mask = torch.arange(T).repeat(B, 1).to(length.device) < length.unsqueeze(-1)
    return mask

def copyComplete(source, target):
    '''https://stackoverflow.com/questions/19787348/copy-file-keep-permissions-and-owner'''
    # copy content, stat-info (mode too), timestamps...
    if os.path.isfile(source):
        shutil.copy2(source, target)
    else:
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
    # copy owner and group
    st = os.stat(source)
    os.chown(target, st.st_uid, st.st_gid)

data_permission = os.access('/data/epinyoan', os.R_OK | os.W_OK | os.X_OK)
base_dir = '/data' if data_permission else '/home'
def init_save_folder(args, copysource=True):
    import glob
    global base_dir
    if args.exp_name != 'TEMP':
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.out_dir = f"./{args.out_dir}/{date}_{args.exp_name}/"
        save_source = f'{args.out_dir}source/'
        os.makedirs(save_source, mode=os.umask(0), exist_ok=True)
    else:
        args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')

def uniform(shape, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def linear_schedule(t):
    return 1 - t

def square_schedule(t):
    return 1 - t**2

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# Example input:
#        [[ 0.3596,  0.0862,  0.9771, -1.0000, -1.0000, -1.0000],
#         [ 0.4141,  0.1781,  0.6628,  0.5721, -1.0000, -1.0000],
#         [ 0.9428,  0.3586,  0.1659,  0.8172,  0.9273, -1.0000]]
# Example output:
#        [[  -inf,   -inf, 0.9771,   -inf,   -inf,   -inf],
#         [  -inf,   -inf, 0.6628,   -inf,   -inf,   -inf],
#         [0.9428,   -inf,   -inf,   -inf,   -inf,   -inf]]
def top_k(logits, thres = 0.9, dim = 1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    # func verified
    # print(probs)
    # print(logits)
    # raise
    return probs


# https://github.com/lucidrains/DALLE-pytorch/issues/318
# https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
from torch.nn import functional as F
def top_p(logits, thres = 0.1):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > (1 - thres)
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

    logits[indices_to_remove] = float('-inf')

    return logits


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch_size, seq_len, dim)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            temperature: scaling factor to modify the logits distribution before applying softmax.
            filter_value: value to set for filtered logits (default: -inf)
    """
    top_k = min(top_k, logits.size(-1))  # Safety check

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        topk_logits, _ = torch.topk(logits, top_k, dim=-1)
        indices_to_remove = logits < topk_logits[..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    
    return logits



def load_trans_from_MMM(trans_encoder, args):
    if args.master_process:
        args.logger.info('loading MMM checkpoint from {}'.format(args.resume_from_MMM))
    
    my_sd = trans_encoder.state_dict()
    mmm_sd = torch.load(os.path.join(args.resume_from_MMM, 'net_last.pth'), map_location='cpu')['trans']

    updated_sd = {}
    for k, v in mmm_sd.items():
        if k in my_sd and my_sd[k].shape == v.shape:
            updated_sd[k] = v

    try:
        my_sd['trans_base.learn_tok_emb.weight'][:2] = mmm_sd['trans_base.learn_tok_emb.weight'][:2]
        my_sd['trans_base.learn_tok_emb.weight'][2:] = mmm_sd['trans_base.learn_tok_emb.weight'][-2:-1]
    except:
        pass

    my_sd.update(updated_sd)
    trans_encoder.load_state_dict(my_sd)
    
    return trans_encoder


def load_last_transformer(transformer, args):
    try:
        if args.master_process:
            args.logger.info('loading checkpoint from {}'.format(args.resume_pth))
        ckpt = torch.load(os.path.join(args.resume_pth, 'net_last.pth'), map_location=args.device)
        print(transformer.load_state_dict(ckpt['trans'], strict=True))
        epoch = ckpt['epoch']
        return transformer, epoch
    except:
        print(50*'--')
        print('Could Not Load Model State Dict')
        print(50*'--')
    return transformer, 0

def load_last_opt_sch(optimizer, scheduler, args):
    try:
        ckpt = torch.load(os.path.join(args.resume_pth, 'net_last.pth'), map_location=args.device)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    except:
        print(50*'--')
        print('Could Not Load Optimizer State Dict')
        print(50*'--')
    return optimizer, scheduler


def save_last_transformer(transformer, optimizer, scheduler, curr_iter, loss, acc, args):
    torch.save(
        {
            'model': transformer.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'curr_iter': curr_iter,
            'loss': loss,
            'acc': acc
        }, os.path.join(args.out_dir, 'transformer_last.pth')
    )

def load_best_transformer(args, save_name='net_best_fid'):
    ckpt = torch.load(os.path.join(args.resume_pth, save_name + '.pth'), map_location='cpu')
    best_loss = ckpt['loss']
    acc = ckpt['acc']
    return best_loss, acc


class SaveBestTransformer:
    def __init__(self, args, best_fid):
        self.best_fid = 1000
        self.save_path = os.path.join(args.out_dir, 'best_loss.pth')

    def save_best(self, transformer, optimizer, loss, acc):
        if loss < self.best_loss:
            self.best_loss = loss

            torch.save({
                'model': transformer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': self.best_loss,
                'acc': acc
            }, self.save_path)


def load_the_most_repeated_codes_to_model_embedding(args, model):
    path = os.path.join(args.codebook_dir, '..')

    if os.path.exists(os.path.join(path, 'max_codes.npy')):
        max_codes = np.load(os.path.join(path, 'max_codes.npy'))
    
    else:
        all_codes = {code_id:0 for code_id in range(args.nb_code)}

        files = os.listdir(args.codebook_dir)
        for f in files:
            target = np.load(os.path.join(args.codebook_dir, f))
            for code_id in target.reshape(-1).tolist():
                all_codes[code_id] += 1
            
        codes = dict(sorted(all_codes.items(), key=lambda item: item[1], reverse=True))
        i = 0
        max_codes = []
        for k, v in codes.items():
            max_codes.append(k)
            i += 1
            if i == 100:
                break
        
        np.save(os.path.join(path, 'max_codes.npy'), np.array(max_codes))

    n_embed_base = model.trans_base.learn_tok_emb.weight.data.shape[0]
    max_codes_idx = torch.tensor(max_codes)[:n_embed_base-2].to(args.device)
    max_codes = model.trans_base.vqvae.vqvae.quantizer.dequantize(max_codes_idx.to(args.device))
    model.trans_base.learn_tok_emb.weight.data[2:] = max_codes

    return model

def load_vq_pretrained(net, args):
    if args.master_process:
        args.logger.info('loading checkpoint from {}'.format(args.vq_pretrained_path))
    ckpt = torch.load(args.vq_pretrained_path, map_location=args.device)
    net.load_state_dict(ckpt['net'], strict=True)
    return net


def load_vq_pretrained_for_vq_transformer(net, args):
    if args.master_process:
        args.logger.info('loading checkpoint from {}'.format(args.vq_pretrained_path))

    my_sd = net.state_dict()
    mmm_sd = torch.load(os.path.join(args.vq_pretrained_path, "net_last.pth"), map_location=args.device)

    updated_sd = {}
    for k, v in mmm_sd.items():
        if k in my_sd and my_sd[k].shape == v.shape:
            updated_sd[k] = v

    my_sd.update(updated_sd)
    net.load_state_dict(my_sd)

    return net



def show(arrays, titles=None, b=0, randperm=None, show_texts=True):
    # Display attention matrix
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    arrays = [arr[b] for arr in arrays]
    if randperm is not None:
        randperm = randperm[b].cpu().numpy()

    arrays = np.array([arr.cpu().numpy() for arr in arrays])

    if titles is None:
        titles = []
        for _ in arrays:
            titles.append('')
    
    # Number of arrays
    num_arrays = len(arrays)
    
    # Create a colormap
    cmap = mcolors.ListedColormap(['red', 'green'])
    
    # Define the plot
    fig, axs = plt.subplots(1, num_arrays, figsize=(5 * num_arrays, 5))
    
    if num_arrays == 1:
        axs = [axs]
    
    for i, (array, title) in enumerate(zip(arrays, titles)):
        # Plot the array
        cax = axs[i].matshow(array, cmap=cmap)
        
        # Set up gridlines
        axs[i].set_xticks(np.arange(-.5, array.shape[1], 1), minor=True)
        axs[i].set_yticks(np.arange(-.5, array.shape[0], 1), minor=True)
        axs[i].grid(which='minor', color='black', linestyle='-', linewidth=2)

        if show_texts:
            # Set labels
            axs[i].set_xticks(np.arange(array.shape[0]))
            axs[i].set_yticks(np.arange(array.shape[0]))

            msg = []
            if randperm is not None:
                for ii in range(array.shape[0]):
                    msg.append(f'{ii:2d}|{randperm[ii]:2d}')
            else:
                msg = list(range(array.shape[0]))
            axs[i].set_xticklabels(msg, rotation='vertical')
            msg = []
            if randperm is not None:
                for ii in range(array.shape[0]):
                    msg.append(f'{randperm[ii]:2d}|{ii:2d}')
            else:
                msg = list(range(array.shape[0]))
            axs[i].set_yticklabels(msg)
            
            # Set title
            axs[i].set_title(title)
    
            # Create a custom legend
            green_patch = mpatches.Patch(color='green', label='Attention allowed')
            red_patch = mpatches.Patch(color='red', label='Attention not allowed')
    
            # Add legend to the figure
            fig.legend(handles=[green_patch, red_patch],)
    
    # Show the plot
    plt.tight_layout()
    plt.show()



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_random_state():
    return {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all(),
        'numpy': np.random.get_state(),
        'random': random.getstate()
    }

# Restore the random state
def restore_random_state(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['random'])


def load_vqvae_from_MMM(net, args):
    if args.master_process:
        args.logger.info('loading checkpoint from {}'.format(args.load_vqvae_from_MMM))
    my_sd = net.state_dict()
    mmm_sd = torch.load(os.path.join(args.load_vqvae_from_MMM, 'net_last.pth'), map_location='cpu')['net']

    updated_sd = {}
    for k, v in mmm_sd.items():
        if k in my_sd and my_sd[k].shape == v.shape:
            updated_sd[k] = v
            if args.master_process:
                print(f"VQ-MMM ---> VQ-BOTH for k={k}")
        else:
            if args.master_process:
                print(f"VQ-MMM |||| VQ-BOTH for k={k}")

    my_sd.update(updated_sd)
    net.load_state_dict(my_sd)
    return net

def load_last_vqvae(net, args):
    pass