import os 
import torch
import numpy as np
import json
import copy
import random
from options import option
import models.vqvae as vqvae
import utils.utils_model as utils_model
from utils.evaluate import evaluation_for_generated_tokens,generate_preference_data
from dataloader.eval_loader import DATALoader
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from param_class import EvaluationParams
# import warnings
# warnings.filterwarnings('ignore')
import sys
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import torch


args = option.get_args_parser()

def main() -> None:

    random.seed(args.seed) 
    torch.manual_seed(args.seed)  
    torch.cuda.manual_seed(args.seed)

    # os.makedirs(args.out_dir, exist_ok = True)

    ##### ---- Logger ---- #####
    # logger = utils_model.get_logger(args.out_dir)
    # logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    from utils.word_vectorizer import WordVectorizer
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    val_loader = DATALoader(args.dataname, args.split, 32, w_vectorizer, unit_length=2**args.down_t,seed=args.seed)

    if args.dataname == 'kit' : 
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'  
        args.nb_joints = 21
    elif args.dataname in ['t2m','idea400']:
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 22

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


    print('Loading VAE')
    vae = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        512,
                        args.code_dim,
                        args.output_emb_width,
                        2,
                        args.stride_t,
                        args.width,
                        3,
                        args.dilation_growth_rate)
    # resume_pth = f"./checkpoints/pretrained_vqvae/{args.dataname}.pth"
    ckpt = torch.load(args.vqvae_path, map_location='cpu')
    vae.load_state_dict(ckpt['net'], strict=True)
    vae = vae.cuda().eval()
    print('Loading VAE Done')


    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    repeat_time = 1

    params = EvaluationParams(
    val_loader=val_loader,
    net=vae,
    eval_wrapper=eval_wrapper,
    start_id = args.start,
    end_id = args.end,
    out_dir = args.out_dir,
    generated_file = args.generated_file,
    )
    for _ in range(repeat_time):
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching= evaluation_for_generated_tokens(params)
        if args.start!=-1:
            return
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)

    print('final result:')
    print('fid: ', sum(fid)/repeat_time)
    print('div: ', sum(div)/repeat_time)
    print('top1: ', sum(top1)/repeat_time)
    print('top2: ', sum(top2)/repeat_time)
    print('top3: ', sum(top3)/repeat_time)
    print('matching: ', sum(matching)/repeat_time)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"
    # logger.info(msg_final)
    print(msg_final)


def format_dpo_file(sft_json, selected_json,dpo_json):


    
    with open(sft_json,'r') as f:
        sft_data = json.load(f)
    with open(selected_json,'r') as f:
        selected_data = json.load(f)
    # with open(gt_json,'r') as f:
    #     gt_data = json.load(f)
    final_data = []
    #selected_data: key:value key is input_text, value is the predicted motions
    for item in tqdm(sft_data):
        temp_dict = {}
        cur_text = item['conversations'][0]['value'].split('Motion description: ')[1]
        conv = item['conversations']
        try:
            chosen_answer = selected_data[cur_text]['chosen']

            # chosen_answer = gt_data[cur_text]
        except:
            continue
        rejected_answer = selected_data[cur_text]['rejected']
        temp_dict.update(item)
        '''
        conv[1]['value'] = chosen_answer
        temp_dict['chosen_conversations'] = conv
        conv[1]['value'] = rejected_answer
        temp_dict['rejected_conversations'] = conv
        '''
        chosen_conv = copy.deepcopy(conv)
        chosen_conv[1]['value'] = chosen_answer
        temp_dict['chosen_conversations'] = chosen_conv

        rejected_conv = copy.deepcopy(conv)
        rejected_conv[1]['value'] = rejected_answer
        temp_dict['rejected_conversations'] = rejected_conv
        final_data.append(temp_dict)

    print(len(final_data))
    with open(dpo_json,'w') as f:
        json.dump(final_data,f)

def dpo_selection():
    random.seed(args.seed) 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    base = "../resources/llm_generated_text"
    candidate_files = []
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith(".json"):
                candidate_files.append(os.path.join(root, f))

    from utils.word_vectorizer import WordVectorizer
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    train_loader = DATALoader(args.dataname, args.split, 32, w_vectorizer, unit_length=2**args.down_t,seed=args.seed)

    if args.dataname == 'kit' : 
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'  
        args.nb_joints = 21
    else :
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 22

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


    print('Loading VAE')
    vae = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        512,
                        args.code_dim,
                        args.output_emb_width,
                        2,
                        args.stride_t,
                        args.width,
                        3,
                        args.dilation_growth_rate)
    ckpt = torch.load(args.vqvae_path, map_location='cpu')
    vae.load_state_dict(ckpt['net'], strict=True)
    vae = vae.cuda().eval()
    print('Loading VAE Done')
    params = EvaluationParams(
    val_loader=train_loader,
    net=vae,
    eval_wrapper=eval_wrapper,
    generated_file = args.generated_file,
    candidate_files = candidate_files,
    fid_weight = args.fid_weight,
    match_weight = args.match_weight,
    )
    generate_preference_data(params)
    format_dpo_file(args.sft_file,args.generated_file,args.dpo_file)
if __name__ == "__main__":
    # warnings.filterwarnings(
    #     # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
    #     "ignore", 
    #     message="ComplexHalf support is experimental and many operators don't support it yet"
    if args.dpo_selection:
        dpo_selection()
    else:    
        main()