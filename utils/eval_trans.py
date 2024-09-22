import os

import clip
import numpy as np
import torch
from scipy import linalg

# import visualization.plot_3d_global as plot_3d
def plot_3d():
    return 0
from utils.motion_process import recover_from_ric
from exit.utils import get_model, visualize_2motions, generate_src_mask
from tqdm import tqdm
import torch.distributed as dist


def all_gather_tensors(input_tensor, args):
    # Check if input_tensor is a NumPy array and convert it to a PyTorch tensor
    if isinstance(input_tensor, np.ndarray):
        local_tensor = torch.tensor(input_tensor,
                                    device=args.device,
                                    dtype=torch.from_numpy(input_tensor).dtype)
    else:
        local_tensor = input_tensor.to(args.device)
    
    gathered_list = [torch.zeros_like(local_tensor) for _ in range(args.world_size)]

    # All-gather operation
    dist.all_gather(gathered_list, local_tensor)

    # Concatenate tensors along the batch dimension
    gathered_tensor = torch.cat(gathered_list, dim=0)

    # Convert the concatenated tensor back to a NumPy array if the input was a NumPy array
    if isinstance(input_tensor, np.ndarray):
        gathered_tensor = gathered_tensor.cpu().numpy().astype(input_tensor.dtype)

    return gathered_tensor


def all_reduce_sum(input_val_or_tensor, args):
    # Check if input_tensor is a NumPy array and convert it to a PyTorch tensor
    if isinstance(input_val_or_tensor, np.ndarray): # A numpy array
        local_tensor = torch.tensor(input_val_or_tensor, 
                                    device=args.device,
                                    dtype=torch.from_numpy(input_val_or_tensor).dtype)
    elif 'numpy' in str(type(input_val_or_tensor)): # a single number with dtype of numpy
        local_tensor = torch.tensor(input_val_or_tensor, device=args.device,
                                    dtype=torch.float64)
    else:
        local_tensor = torch.tensor(input_val_or_tensor).to(args.device)
    
    # Perform the all-reduce operation to sum across all processes
    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)

    if isinstance(input_val_or_tensor, np.ndarray):
        summed_val_or_tensor = local_tensor.cpu().numpy().astype(np.float64)
    elif 'numpy' in str(type(input_val_or_tensor)):
        summed_val_or_tensor = np.float64(local_tensor.item())
    else:
        summed_val_or_tensor = local_tensor.item()

    return summed_val_or_tensor


def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname)
    plot_xyz =np.transpose(plot_xyz, (0, 1, 4, 2, 3)) 
    writer.add_video(tag, plot_xyz, nb_iter, fps = 20)




@torch.no_grad()        
def evaluation_vqvae(out_dir, val_loader, net, nb_iter,
                     best_fid, best_iter, best_div, best_top1, best_top2,
                     best_top3, best_matching, eval_wrapper, optimizer=None,
                     scheduler=None, draw = True, dataname=None,
                     save = True, savegif=False, savenpy=False,
                     text_embedding=None, args=None):
    
    try:    net = net.module
    except: net = net

    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []


    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            # pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


            pred_pose, loss_commit, perplexity = net(motion[i:i+1, :m_length[i]], dataname=dataname)
            # pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            # pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
            
            # if savenpy:
            #     np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
            #     np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose

            # if i < min(4, bs):
            #     draw_org.append(pose_xyz)
            #     draw_pred.append(pred_xyz)
            #     draw_text.append(caption[i])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    args.logger.info(50*'---')
    args.logger.info(dataname)
    args.logger.info(msg)
    
    # if draw:
    args.writer.add_scalar('./Test/FID', fid, nb_iter)
    args.writer.add_scalar('./Test/Diversity', diversity, nb_iter)
    args.writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
    args.writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
    args.writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
    args.writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        # if nb_iter % 5000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        # if nb_iter % 5000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)   

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        args.logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, f'{dataname}_net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        args.logger.info(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        args.logger.info(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        args.logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        args.logger.info(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        args.logger.info(msg)
        best_matching = matching_score_pred
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if save:
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching


@torch.no_grad()        
def evaluation_transformer(out_dir, val_loader, net, trans, nb_iter, best_fid, 
                           best_iter, best_div, best_top1, best_top2, best_top3, 
                           best_matching, clip_model, eval_wrapper, dataname='t2m', 
                           draw = True, save = True, savegif=False, num_repeat=1, rand_pos=False, 
                           CFG=-1, text_embedding=None, args=None, optimizer=None, scheduler=None, 
                           epoch=None):
    args.logger.info(20*'-----')
    args.logger.info(f"Rank {args.rank} - Start Validation...")
    
    try:    raw_trans = trans.module
    except: raw_trans = trans

    if num_repeat < 0:
        is_avg_all = True
        num_repeat = -num_repeat
    else:
        is_avg_all = False


    trans.eval(), net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = np.float32(0)
    R_precision = np.float32(0)
    matching_score_real = np.float32(0)
    matching_score_pred = np.float32(0)

    nb_sample = np.float32(0)
    blank_id = get_model(raw_trans).num_vq

    for batch in tqdm(val_loader, desc=f"Rank: {args.rank} - {raw_trans.sample.__name__}"):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

        feat_clip_text, word_emb = text_embedding(clip_text)

        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        motion_multimodality_batch = []
        m_tokens_len = torch.ceil((m_length)/4)

        pred_len = m_length.to(args.device)
        pred_tok_len = m_tokens_len

        for i in range(num_repeat):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(args.device)
            # pred_len = torch.ones(bs).long()

            index_motion = raw_trans(feat_clip_text, word_emb, type="sample", m_length=pred_len,
                                     CFG=CFG, max_steps=args.max_steps_generation)
            # [INFO] 1. this get the last index of blank_id
            # pred_length = (index_motion == blank_id).int().argmax(1).float()
            # [INFO] 2. this get the first index of blank_id
            pred_length = (index_motion >= blank_id).int()
            pred_length = torch.topk(pred_length, k=1, dim=1).indices.squeeze().float()
            # pred_length[pred_length==0] = index_motion.shape[1] # if blank_id in the first frame, set length to max
            # [INFO] need to run single sample at a time b/c it's conv
            for k in range(bs):
                m = index_motion[k:k+1, :int(pred_tok_len[k].item())]
                pred_pose = net(m, type='decode')
                pred_pose_eval[k:k+1, :int(pred_len[k].item())] = pred_pose

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)
            ######################################################
            et_pred, em_pred = et_pred.float(), em_pred.float()
            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0 or is_avg_all:
                pose = pose.to(args.device).float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                et, em = et.float(), em.float()
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                # if draw:
                #     pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                #     pose_xyz = recover_from_ric(torch.from_numpy(pose).float().to(args.device), num_joints)

                #     for j in range(min(4, bs)):
                #         draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                #         draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                R_precision_real = np.float32(R_precision_real)
                matching_score_real = np.float32(matching_score_real)

                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match
                R_precision = np.float32(R_precision)
                matching_score_real = np.float32(matching_score_real)

                nb_sample += bs
        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    if args.ddp_eval:
        motion_annotation_np = all_gather_tensors(motion_annotation_np, args)
        motion_pred_np = all_gather_tensors(motion_pred_np, args)

    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100, args.debug)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100, args.debug)

    if args.ddp_eval:
        R_precision_real = all_reduce_sum(R_precision_real, args)
        R_precision = all_reduce_sum(R_precision, args)
        nb_sample = all_reduce_sum(nb_sample, args)

        matching_score_real = all_reduce_sum(matching_score_real, args)
        matching_score_pred = all_reduce_sum(matching_score_pred, args)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    R_precision_real = np.float32(R_precision_real)
    matching_score_real = np.float32(matching_score_real)

    R_precision = np.float32(R_precision)
    matching_score_real = np.float32(matching_score_real)

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu()
    if num_repeat > 10:
        if args.ddp_eval:
            motion_multimodality = all_gather_tensors(motion_multimodality, args)
        multimodality = calculate_multimodality(motion_multimodality, 10, args.debug)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    sampling_type_ = raw_trans.sample.__name__

    # if args.master_process:
    msg = f"--> \t {sampling_type_} | Eva. Iter {nb_iter} :, \n\
                FID. {fid:.4f} , \n\
                Diversity Real. {diversity_real:.4f}, \n\
                Diversity. {diversity:.4f}, \n\
                R_precision_real. {R_precision_real}, \n\
                R_precision. {R_precision}, \n\
                matching_score_real. {matching_score_real}, \n\
                matching_score_pred. {matching_score_pred}, \n\
                multimodality. {multimodality:.4f}"
    
    args.logger.info(msg)
    
    if args.rank == 0:
        if draw:
            args.writer.add_scalar(f'./Test/{sampling_type_}/FID', fid, nb_iter)
            args.writer.add_scalar(f'./Test/{sampling_type_}/Diversity', diversity, nb_iter)
            args.writer.add_scalar(f'./Test/{sampling_type_}/top1', R_precision[0], nb_iter)
            args.writer.add_scalar(f'./Test/{sampling_type_}/top2', R_precision[1], nb_iter)
            args.writer.add_scalar(f'./Test/{sampling_type_}/top3', R_precision[2], nb_iter)
            args.writer.add_scalar(f'./Test/{sampling_type_}/matching_score', matching_score_pred, nb_iter)
            args.writer.add_scalar(f'./Test/{sampling_type_}/multimodality', multimodality, nb_iter)

            # if nb_iter % 10000 == 0 : 
            #     for ii in range(4):
            #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            # if nb_iter % 10000 == 0 : 
            #     for ii in range(4):
            #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

        
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        args.logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        # if save and not args.debug:
        #     torch.save({'trans' : raw_trans.state_dict()}, os.path.join(out_dir, f'{sampling_type_}_{start_ids_}_from_T_{sampling_from_T_}_net_best_fid.pth'))
    
    if matching_score_pred < best_matching : 
        # msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        # args.logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        # msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        # args.logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1 : 
        # msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        # args.logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2 : 
        # msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        # args.logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        # msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        # args.logger.info(msg)
        best_top3 = R_precision[2]

    if save and args.rank == 0 and not args.debug:
        torch.save({'trans' : raw_trans.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch}, 
                    os.path.join(out_dir, 'net_last.pth'))
    
    args.logger.info(f"Rank {args.rank} - Finished Validation.")
    args.logger.info(20*'-----')

    raw_trans.train()
    return fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, multimodality 

def evaluation_transformer_uplow(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, dataname, draw = True, save = True, savegif=False, num_repeat=1, rand_pos=False, CFG=-1) : 
    from utils.humanml_utils import HML_UPPER_BODY_MASK, HML_LOWER_BODY_MASK

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    blank_id = get_model(trans).num_vq
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        pose = pose.cuda().float()
        pose_lower = pose[..., HML_LOWER_BODY_MASK]
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text, word_emb = clip_model(text)
        
        motion_multimodality_batch = []
        m_tokens_len = torch.ceil((m_length)/4)

         
        pred_len = m_length.cuda()
        pred_tok_len = m_tokens_len

        max_motion_length = int(seq/4) + 1
        mot_end_idx = get_model(net).vqvae.num_code
        mot_pad_idx = get_model(net).vqvae.num_code + 1
        target_lower = []
        for k in range(bs):
            target = net(pose[k:k+1, :m_length[k]], type='encode')
            if m_tokens_len[k]+1 < max_motion_length:
                target = torch.cat([target, 
                                    torch.ones((1, 1, 2), dtype=int, device=target.device) * mot_end_idx, 
                                    torch.ones((1, max_motion_length-1-m_tokens_len[k].int().item(), 2), dtype=int, device=target.device) * mot_pad_idx], axis=1)
            else:
                target = torch.cat([target, 
                                    torch.ones((1, 1, 2), dtype=int, device=target.device) * mot_end_idx], axis=1)
            target_lower.append(target[..., 1])
        target_lower = torch.cat(target_lower, axis=0)

        for i in range(num_repeat):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            # pred_len = torch.ones(bs).long()

            index_motion = trans(feat_clip_text, target_lower, word_emb, type="sample", m_length=pred_len, rand_pos=rand_pos, CFG=CFG)
            # [INFO] 1. this get the last index of blank_id
            # pred_length = (index_motion == blank_id).int().argmax(1).float()
            # [INFO] 2. this get the first index of blank_id
            pred_length = (index_motion >= blank_id).int()
            pred_length = torch.topk(pred_length, k=1, dim=1).indices.squeeze().float()
            # pred_length[pred_length==0] = index_motion.shape[1] # if blank_id in the first frame, set length to max
            # [INFO] need to run single sample at a time b/c it's conv
            for k in range(bs):
            ######### [INFO] Eval only the predicted length
            #     if pred_length[k] == 0:
            #         pred_len[k] = seq
            #         continue
            #     pred_pose = net(index_motion[k:k+1, :int(pred_length[k].item())], type='decode')
            #     cur_len = pred_pose.shape[1]

            #     pred_len[k] = min(cur_len, seq)
            #     pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
            # et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            ######################################################
            
            ######### [INFO] Eval by m_length
                all_tokens = torch.cat([
                    index_motion[k:k+1, :int(pred_tok_len[k].item()), None],
                    target_lower[k:k+1, :int(pred_tok_len[k].item()), None]
                ], axis=-1)
                pred_pose = net(all_tokens, type='decode')
                pred_pose_eval[k:k+1, :int(pred_len[k].item())] = pred_pose
            pred_pose_eval[..., HML_LOWER_BODY_MASK] = pose_lower
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)
            ######################################################

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                # if draw:
                #     pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                #     pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


                #     for j in range(min(4, bs)):
                #         draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                #         draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs
        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    if num_repeat > 1:
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, \n\
                FID. {fid:.4f} , \n\
                Diversity Real. {diversity_real:.4f}, \n\
                Diversity. {diversity:.4f}, \n\
                R_precision_real. {R_precision_real}, \n\
                R_precision. {R_precision}, \n\
                matching_score_real. {matching_score_real}, \n\
                matching_score_pred. {matching_score_pred}, \n\
                multimodality. {multimodality:.4f}"
    logger.info(msg)
    
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)
        writer.add_scalar('./Test/multimodality', multimodality, nb_iter)

        # if nb_iter % 10000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
        # if nb_iter % 10000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    
    if fid < best_fid : 
        # msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        # if save:
        #     torch.save({'trans' : get_model(trans).state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
    
    if matching_score_pred < best_matching : 
        # msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        # msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1 : 
        # msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2 : 
        # msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        # msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'trans' : get_model(trans).state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, writer, logger

@torch.no_grad()        
def evaluation_transformer_test(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    
    for batch in val_loader:

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(30):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()
            
            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k:k+1], True)
                except:
                    index_motion = torch.ones(1,1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)
    
    
    if draw:
        for ii in range(len(draw_org)):
            tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)
        
            tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times, debug=False):
    assert len(activation.shape) == 3
    if debug: multimodality_times = activation.shape[1] - 1
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times, debug=False):
    assert len(activation.shape) == 2
    if debug: diversity_times = activation.shape[0] - 1
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist