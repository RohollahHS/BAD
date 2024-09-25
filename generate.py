import os
import random
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import clip
import models.vqvae as vqvae
from models.vqvae_sep import VQVAE_SEP
import models.t2m_trans as trans
import models.t2m_trans_uplow as trans_uplow
import numpy as np
from exit.utils import visualize_2motions, set_seed, seed_worker
import options.option_transformer as option_trans
from models.clip_model import load_clip_model
import my_clip
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
from os.path import join as pjoin
from visualization.joints2bvh import Joint2BVHConvertor
import time
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

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


def text_embedding(clip_text):
    text = my_clip.tokenize(clip_text, truncate=True).to(args.device)
    feat_clip_text, word_emb = clip_model.encode_text(text)
    return feat_clip_text.float(), word_emb.float()


def get_vqvae(args, is_upper_edit):
    if not is_upper_edit:
        return vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                            args.nb_code,
                            args.code_dim,
                            args.output_emb_width,
                            args.down_t,
                            args.stride_t,
                            args.width,
                            args.depth,
                            args.dilation_growth_rate)
    else:
        return VQVAE_SEP(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate,
                        moment={'mean': torch.from_numpy(args.mean).to(args.device).float(), 
                            'std': torch.from_numpy(args.std).to(args.device).float()},
                        sep_decoder=True)

def get_maskdecoder(args, vqvae, is_upper_edit):
    tranformer = trans if not is_upper_edit else trans_uplow
    return tranformer.Text2Motion_Transformer(vqvae,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate,
                                args=args)

class BAD(torch.nn.Module):
    def __init__(self, args=None, is_upper_edit=False):
        super().__init__()
        self.is_upper_edit = is_upper_edit
        self.args = args

        args.dataname = args.dataset_name = 't2m'

        self.vqvae = get_vqvae(args, is_upper_edit)
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        self.vqvae.load_state_dict(ckpt['net'], strict=True)
        if is_upper_edit:
            class VQVAE_WRAPPER(torch.nn.Module):
                def __init__(self, vqvae) :
                    super().__init__()
                    self.vqvae = vqvae
                    
                def forward(self, *args, **kwargs):
                    return self.vqvae(*args, **kwargs)
            self.vqvae = VQVAE_WRAPPER(self.vqvae)
        self.vqvae.eval()
        self.vqvae.to(args.device)

        self.maskdecoder = get_maskdecoder(args, self.vqvae, is_upper_edit)
        ckpt = torch.load(args.resume_trans, map_location='cpu')
        self.maskdecoder.load_state_dict(ckpt['trans'], strict=True)
        self.maskdecoder.train()
        self.maskdecoder.to(args.device)

    def forward(self, text, lengths=-1):
        feat_clip_text, word_emb = text_embedding(text)
    
        if lengths == -1:
            print("Since no motion length are specified, we will use estimated motion lengthes!!")
            pred_dis = length_estimator(feat_clip_text)
            probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
            m_token_length = torch.distributions.Categorical(probs).sample()  # (b, seqlen)
            lengths = m_token_length * 4
            # lengths = torch.multinomial()
        else:
            if not isinstance(lengths, list):
                lengths = [lengths]
            lengths = torch.tensor(lengths).to(self.args.device)
            m_token_length = torch.ceil((lengths)/4).long()
            lengths = m_token_length * 4

        index_motion = self.maskdecoder(feat_clip_text, word_emb, type="sample", m_length=lengths, if_test=False)

        B = len(index_motion)
        pred_pose_all = torch.zeros((B, 196, 263)).to(args.device)
        for k in range(B):
            pred_pose = self.vqvae(index_motion[k:k+1, :m_token_length[k]], type='decode')
            pred_pose_all[k:k+1, :int(lengths[k].item())] = pred_pose
        return pred_pose_all, lengths

    def call_T2MBD(self, inbetween_text, pose, m_length):
        ### FOR NO TEST ###:
        edit_task = self.args.edit_task
        feat_clip_text, word_emb = text_embedding(inbetween_text)

        bs, seq = pose.shape[:2]
        tokens = -1*torch.ones((bs, 50), dtype=torch.long).cuda()

        if edit_task in ['inbetween', 'outpainting']:
            m_token_length = torch.ceil((m_length)/4).int().cpu().numpy()
            m_token_length_init = (m_token_length * .25).astype(int)
            m_length_init = (m_length * .25).int()
            for k in range(bs):
                l = m_length_init[k]
                l_token = m_token_length_init[k]

                if edit_task == 'inbetween':
                    # start tokens
                    index_motion = self.vqvae(pose[k:k+1, :l].cuda(), type='encode')
                    tokens[k,:index_motion.shape[1]] = index_motion[0]

                    # end tokens
                    index_motion = self.vqvae(pose[k:k+1, m_length[k]-l :m_length[k]].cuda(), type='encode')
                    tokens[k, m_token_length[k]-l_token :m_token_length[k]] = index_motion[0]
                elif edit_task == 'outpainting':
                    # inside tokens
                    index_motion = self.vqvae(pose[k:k+1, l:m_length[k]-l].cuda(), type='encode')
                    tokens[k, l_token: l_token+index_motion.shape[1]] = index_motion[0]

        if edit_task in ['prefix', 'suffix']:
            m_token_length = torch.ceil((m_length)/4).int().cpu().numpy()
            m_token_length_half = (m_token_length * .5).astype(int)
            m_length_half = (m_length * .5).int()
            for k in range(bs):
                if edit_task == 'prefix':
                    index_motion = self.vqvae(pose[k:k+1, :m_length_half[k]].cuda(), type='encode')
                    tokens[k, :m_token_length_half[k]] = index_motion[0]
                elif edit_task == 'suffix':
                    index_motion = self.vqvae(pose[k:k+1, m_length_half[k]:m_length[k]].cuda(), type='encode')
                    tokens[k, m_token_length[k]-m_token_length_half[k] :m_token_length[k]] = index_motion[0]

        inpaint_index = self.maskdecoder(feat_clip_text, word_emb, type="sample",
                                         m_length=m_length.cuda(), token_cond=tokens)

        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
        for k in range(bs):
            pred_pose = self.vqvae(inpaint_index[k:k+1, :m_token_length[k]], type='decode')
            pred_pose_eval[k:k+1, :int(m_length[k].item())] = pred_pose
        
        return pred_pose_eval


    def temporal_editing_preprocess(self, pose, m_length):
        edit_task = args.edit_task
        bs, seq = pose.shape[:2]
        tokens = -1*torch.ones((bs, 50), dtype=torch.long).cuda()

        if edit_task in ['inbetween', 'outpainting']:
            m_token_length = torch.ceil((m_length)/4).int().cpu().numpy()
            m_token_length_init = (m_token_length * .25).astype(int)
            m_length_init = (m_length * .25).int()
            for k in range(bs):
                l = m_length_init[k]
                l_token = m_token_length_init[k]

                if edit_task == 'inbetween':
                    # start tokens
                    index_motion = self.vqvae(pose[k:k+1, :l].cuda(), type='encode')
                    tokens[k,:index_motion.shape[1]] = index_motion[0]

                    # end tokens
                    index_motion = self.vqvae(pose[k:k+1, m_length[k]-l :m_length[k]].cuda(), type='encode')
                    tokens[k, m_token_length[k]-l_token :m_token_length[k]] = index_motion[0]
                elif edit_task == 'outpainting':
                    # inside tokens
                    index_motion = self.vqvae(pose[k:k+1, l:m_length[k]-l].cuda(), type='encode')
                    tokens[k, l_token: l_token+index_motion.shape[1]] = index_motion[0]

        if edit_task in ['prefix', 'suffix']:
            m_token_length = torch.ceil((m_length)/4).int().cpu().numpy()
            m_token_length_half = (m_token_length * .5).astype(int)
            m_length_half = (m_length * .5).int()
            for k in range(bs):
                if edit_task == 'prefix':
                    index_motion = self.vqvae(pose[k:k+1, :m_length_half[k]].cuda(), type='encode')
                    tokens[k, :m_token_length_half[k]] = index_motion[0]
                elif edit_task == 'suffix':
                    index_motion = self.vqvae(pose[k:k+1, m_length_half[k]:m_length[k]].cuda(), type='encode')
                    tokens[k, m_token_length[k]-m_token_length_half[k] :m_token_length[k]] = index_motion[0]
        
        return tokens


    def long_range(self, text, lengths, num_transition_token=2, output='concat', index_motion=None):
        assert isinstance(text, list)
        b = len(text)
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths).to(args.device)
        feat_clip_text, word_emb = text_embedding(text)
        if index_motion is None:
            index_motion = self.maskdecoder(feat_clip_text, word_emb, type="sample", m_length=lengths)

        m_token_length = torch.ceil((lengths)/4).int()
        half_token_length = (m_token_length/2).int()
        idx_full_len = half_token_length >= 24
        half_token_length[idx_full_len] = half_token_length[idx_full_len] - 1

        mask_id = self.maskdecoder.num_vq + 2
        tokens = -1*torch.ones((b-1, 50), dtype=torch.long).to(args.device)
        transition_train_length = []
        
        for i in range(b-1):
            i_index_motion = index_motion[i]
            i1_index_motion = index_motion[i+1]
            left_end = half_token_length[i]

            right_start = left_end + num_transition_token
            end = right_start + half_token_length[i+1]

            tokens[i, :left_end] = i_index_motion[m_token_length[i]-left_end: m_token_length[i]]
            tokens[i, left_end:right_start] = mask_id
            tokens[i, right_start:end] = i1_index_motion[:half_token_length[i+1]]
            transition_train_length.append(end)

        transition_train_length = torch.tensor(transition_train_length).to(index_motion.device)
        feat_clip_text, word_emb_clip = text_embedding(text[:-1])
        inpaint_index = self.maskdecoder(feat_clip_text, word_emb_clip, type="sample", m_length=transition_train_length*4, token_cond=tokens, max_steps=1)
        
        all_tokens = []
        for i in range(b-1):
            all_tokens.append(index_motion[i, :m_token_length[i]])
            all_tokens.append(inpaint_index[i, tokens[i] == mask_id])
        all_tokens.append(index_motion[-1, :m_token_length[-1]])
        all_tokens = torch.cat(all_tokens).unsqueeze(0)
        pred_pose = self.vqvae(all_tokens, type='decode')
        return pred_pose


if __name__ == '__main__':
    args = option_trans.get_args_parser()
    set_seed(args.seed)

    std = np.load('./exit/t2m-std.npy')
    mean = np.load('./exit/t2m-mean.npy')

    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()
    
    if not args.temporal_editing :
        animation_dir = pjoin('output', 'visualization', 'animation')
        joints_dir = pjoin('output', 'visualization', 'joints')
    else:
        animation_dir = pjoin('output', 'visualization', 'animation_inbetween', args.edit_task)
        joints_dir = pjoin('output', 'visualization', 'joints_inbetween', args.edit_task)
    
    os.makedirs(animation_dir, exist_ok=True)
    os.makedirs(joints_dir, exist_ok=True)

    clip_model = load_clip_model(args).to(args.device)
    length_estimator = load_len_estimator(args).to(args.device).eval()
    bad = BAD(args).to(args.device).eval()

    captions = [args.caption]
    caption_inbetween = [args.caption_inbetween]

    for r in range(args.repeat_times_generation):
        with torch.no_grad():
            print("--> Repeat %d"%r)

            if not args.temporal_editing and not args.long_seq_generation:
                pred_motions, m_length = bad(captions, args.length)
            elif args.long_seq_generation:
                caption = args.long_seq_captions
                m_length = torch.tensor(args.long_seq_lengths).to(args.device)
                pred_motions = bad.long_range(caption, m_length)
                captions = [' | '.join(caption)]
                m_length = torch.tensor([pred_motions.shape[1]])
            else:
                assert args.caption_inbetween is not None
                pose_base, m_length = bad(captions, args.length)
                pred_motions = bad.call_T2MBD(caption_inbetween, pose_base, m_length)
            
            data = pred_motions.detach().cpu().numpy() * std + mean
            m_length = m_length.detach().cpu().numpy()

        for k, (caption, joint_data)  in enumerate(zip(captions, data)):
            print("----> Sample %d: %s %d"%(k, caption, m_length[k]))

            caption_name = caption[:-1].replace(' ', '_')[:30]

            animation_path = pjoin(animation_dir, caption_name+"_"+str(k))
            joint_path = pjoin(joints_dir, caption_name+"_"+str(k))

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

            # bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.bvh"%(k, r, m_length[k]))
            # _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
            _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

            save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4"%(k, r, m_length[k]))
            ik_save_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.mp4"%(k, r, m_length[k]))

            if args.temporal_editing :
                base_caption = caption
                caption_inbetween = caption_inbetween[k]
                caption = f'{args.edit_task}\n=====\n{base_caption}\n=====\n{caption_inbetween}'

            # plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
            plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])), joint)
            # np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_ik.npy"%(k, r, m_length[k])), ik_joint)

            print('animation_path:', save_path)
            print('joint_path:', pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])))

            print()
