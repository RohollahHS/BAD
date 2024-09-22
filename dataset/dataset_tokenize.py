import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm



class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, window_size = 64, unit_length = 8, fill_max_len=False, args=None):
        self.window_size = window_size
        self.unit_length = unit_length
        self.feat_bias = feat_bias
        self.fill_max_len = fill_max_len

        self.dataset_name = dataset_name
        min_motion_len = 40 if dataset_name =='t2m' else 24
        
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            self.dim_pose = 263
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            #kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            self.dim_pose = 251
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            #kinematic_chain = paramUtil.kit_kinematic_chain
        
        joints_num = self.joints_num

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        
        split_file = pjoin(self.data_root, 'train.txt')
        
        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        i_debug = 0
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                data_dict[name] = {'motion': motion,
                                   'length': len(motion),
                                   'name': name}
                new_name_list.append(name)
                length_list.append(len(motion))
                
                if args.debug:
                    i_debug += 1
                    if i_debug >= args.maxdata:
                        break
            except:
                # Some motion may not exist in KIT dataset
                pass


        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        if self.fill_max_len:
            motion_zero = np.zeros((self.max_motion_length, self.dim_pose))
            motion_zero[:m_length] = motion
            motion = motion_zero
            motion = (motion - self.mean) / self.std
            return motion, m_length

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name

def DATALoader(dataset_name,
                batch_size = 1,
                num_workers = 8, unit_length = 4, shuffle=True, args=None) : 
    
    train_loader = torch.utils.data.DataLoader(VQMotionDataset(dataset_name, unit_length=unit_length, fill_max_len=batch_size!=1, args=args),
                                              batch_size,
                                              shuffle=shuffle,
                                              num_workers=0,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def save_tokens(args, net):
    print("Extracting Code")
    train_loader_token = DATALoader(args.dataname, 1, unit_length=2**args.down_t, args=args)
    for batch in tqdm(train_loader_token, "Starting Extracting..."):
        pose, name = batch
        bs, seq = pose.shape[0], pose.shape[1]

        pose = pose.to(args.device).float() # bs, nb_joints, joints_dim, seq_len
        target = net(pose, type='encode')
        target = target.cpu().numpy()
        
        np.save(pjoin(args.codebook_dir, name[0] +'.npy'), target)
