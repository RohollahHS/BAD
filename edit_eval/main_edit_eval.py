import torch
from tqdm import tqdm
import utils.eval_trans as eval_trans

import numpy as np
import json

import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_eval
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from exit.utils import base_dir, init_save_folder, set_seed, seed_worker
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def eval_inbetween(eval_wrapper, logger, val_loader, call_model, nb_iter):
    num_repeat = 1
    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    nb_sample = 0
    
    count = 0
    for word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name in tqdm(val_loader):
        bs, seq = pose.shape[:2]
        motion_multimodality_batch = []
        for i in range(num_repeat):
            pred_pose_eval = call_model(clip_text, pose, m_length)
            
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)
            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0:
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                temp_R, temp_match = eval_trans.calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = eval_trans.calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs
                ### end if
            ### end for
        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))
        # if count > 2:
        #     break
        count += 1
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = eval_trans.calculate_activation_statistics(motion_annotation_np)
    mu, cov= eval_trans.calculate_activation_statistics(motion_pred_np)

    diversity_real = eval_trans.calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = eval_trans.calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    if num_repeat > 1:
        multimodality = eval_trans.calculate_multimodality(motion_multimodality, 10)

    fid = eval_trans.calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

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
    return fid, diversity, R_precision, matching_score_pred, multimodality

def run_all_eval(call_model, out_dir, exp_name, copysource=True, args_orig=None):
    from tqdm import tqdm
    import os
    out_dir = f'{out_dir}/eval_edit'
    os.makedirs(out_dir, exist_ok = True)

    class Temp:
        def __init__(self):
            print('mock:: opt')
    args = Temp() 
    args.out_dir = out_dir
    args.exp_name = exp_name
    args.seed = 123
    args.debug = args_orig.debug
    args.maxdata = args_orig.maxdata
    init_save_folder(args, copysource)

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir, args=args)
    logger.info(json.dumps(vars(args_orig), indent=4, sort_keys=True))
    logger.info(50*'===')
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    from utils.word_vectorizer import WordVectorizer
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    val_dataset = dataset_TM_eval.T2MDataset('t2m', True, 32, w_vectorizer, args=args)

    def get_val_loader():
        g = torch.Generator()
        g.manual_seed(args.seed)
        return torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=32,
                    shuffle=True,
                    num_workers=8 if 'shahab' in os.getcwd() else 0,
                    collate_fn=collate_fn,
                    drop_last=True,
                    pin_memory=True if 'shahab' in os.getcwd() else False,
                    worker_init_fn=seed_worker, generator=g)

    val_loader = get_val_loader()

    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    multi = []
    repeat_time = 20

    for i in tqdm(range(repeat_time)):
        _fid, diversity, R_precision, matching_score_pred, multimodality = eval_inbetween(eval_wrapper, logger, val_loader, call_model, nb_iter=i)
        
        fid.append(_fid)
        div.append(diversity)
        top1.append(R_precision[0])
        top2.append(R_precision[1])
        top3.append(R_precision[2])
        matching.append(matching_score_pred)
        multi.append(multimodality)

    print('final result:')
    print('fid: ', sum(fid)/repeat_time)
    print('div: ', sum(div)/repeat_time)
    print('top1: ', sum(top1)/repeat_time)
    print('top2: ', sum(top2)/repeat_time)
    print('top3: ', sum(top3)/repeat_time)
    print('matching: ', sum(matching)/repeat_time)
    print('multi: ', sum(multi)/repeat_time)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    multi = np.array(multi)
    msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
    logger.info(msg_final)