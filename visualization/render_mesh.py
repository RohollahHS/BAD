import argparse
import os
import sys
sys.path.append(os.getcwd())
from visualization import vis_utils
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--npy_path", type=str)
    parser.add_argument("--n_iter_smpl", type=int, default=150)
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=str, default='cuda', help='')
    params = parser.parse_args()

    assert params.input_path.endswith('.mp4')
    parsed_name = os.path.basename(params.input_path).replace('.mp4', '').replace('sample', '').replace('repeat', '')
    # sample_i, rep_i = [int(e) for e in parsed_name.split('_')[:2]]
    sample_i, rep_i = 0, 0
    if params.npy_path is None:
        npy_path = f"/output/visualization/joints/{params.input_path.split('/')[-1][:-4]}.npy"
    else:
        npy_path = params.npy_path
    # npy_path = os.path.join(os.path.dirname(params.input_path), 'results.npy')
    out_npy_path = params.input_path.replace('.mp4', '_smpl_params.npy')
    assert os.path.exists(npy_path)
    results_dir = params.input_path.replace('.mp4', '_obj')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                device=params.device, cuda=params.cuda, n_iter=params.n_iter_smpl)

    print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
    for frame_i in tqdm(range(npy2obj.real_num_frames)):
        npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)

    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
    npy2obj.save_npy(out_npy_path)