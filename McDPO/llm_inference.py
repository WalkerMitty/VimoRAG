import os 
import torch
import numpy as np
import json
import random
from options import option
# import models.vqvae as vqvae
import utils.utils_model as utils_model
from utils.evaluate import llm_inference_func,llm_inference_demo
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

import torch
from transformers import set_seed
from mcdpo.model.builder import load_pretrained_model,merge_peft_model
from mcdpo.mm_utils import tokenizer_image_token, get_model_name_from_path
args = option.get_args_parser()

def main() -> None:

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) 

    os.makedirs(args.out_dir, exist_ok = True)
    generated_file = os.path.join(args.out_dir, f'start{str(args.start)}.json')
    # if os.path.exists(generated_file):
        # print('file exists, skip')
        # return
    ##### ---- Logger ---- #####
    # logger = utils_model.get_logger(args.out_dir)
    # logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    if not args.demo_inference:
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
    else:
        val_loader = None
        eval_wrapper = None


    set_seed(args.llm_seed)
    print('loading LLMs...')
    model_path = args.model_path
    # Load pretrained model and tokenizer
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    if args.lora:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, args.model_base, model_name
        )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path,None, model_name,lora_weights=False
        )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    # import pdb;pdb.set_trace()
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))
    if args.text_only == False:
        # Load vision towers
        vision_tower = model.get_vision_tower()
        vision_tower.load_model(model.config.mm_vision_tower)
        video_processor = vision_tower.image_processor

        image_vision_tower = model.get_image_vision_tower()
        image_vision_tower.load_model()
        image_processor = image_vision_tower.image_processor
        if args.motion_encoder:
            motion_tower = model.get_motion_tower()
            motion_tower.load_model(model.config.motion_mm_tower)

    else:
        video_processor = None
    # Move model to GPU
    model = model.to("cuda")
    model.eval()
    # import ipdb;ipdb.set_trace()
    params = EvaluationParams(
    val_loader=val_loader,
    # net=vae,
    model=model,
    tokenizer=tokenizer,
    tokenizer_image_token=tokenizer_image_token,
    eval_wrapper=eval_wrapper,
    temperature=args.temperature,
    # video_dir=args.video_path,
    image_processor=image_processor,
    video_processor=video_processor,
    start_id = args.start,
    end_id = args.end,
    # out_dir = args.out_dir,
    retrieval_result = args.retrieval_result,
    text_only = args.text_only,
    generated_file = generated_file,
    motion_encoder = args.motion_encoder,

    )
    
    if args.demo_inference:
        llm_inference_demo(params)
    else:
        llm_inference_func(params)
 
def merge_lora_weights():
    # model_base = "../resources/playground/Phi-3-mini-4k-instruct"
    model_base = args.model_base
    model_path = args.model_path
    merged_model = args.out_dir
    # Load pretrained model and tokenizer
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    merge_peft_model(model_path, model_base, model_name,merged_model)


if __name__ == "__main__":
    if args.merge_lora:
        merge_lora_weights()
    else:
        main()