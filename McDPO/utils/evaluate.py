import os
import re
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
from mcdpo.constants import *
from .losses import ReConsLoss
from .motion_process import recover_from_ric
from param_class import EvaluationParams
from mcdpo.conversation import conv_templates
from eval.video_encoding import _get_rawvideo_dec
import json
from scipy.special import softmax
import time
import sys
def _get_motion_motionbert(cache_keypoints,video_paths,max_frames):
    all_keypoints = []
    for i,video_file_path in enumerate(video_paths):
        try:
            cur_keypoints = cache_keypoints[video_file_path] #[n,17,3]
        except:
            print('cache_keypoints read fails, loading the default one')
            cur_keypoints = cache_keypoints['../resources/wild_motion_videos/aslan/7-0005.mp4']
        

        if cur_keypoints.shape[0]< max_frames:
            padding_num = max_frames - cur_keypoints.shape[0]
            temp = np.zeros((padding_num, 17, 3), dtype=cur_keypoints.dtype)
            cur_keypoints = np.concatenate((cur_keypoints, temp), axis=0)


        all_keypoints.append(cur_keypoints)
        

    return torch.from_numpy(np.array(all_keypoints))

def get_prompt(image_token,qs):
    PROMPT = f'Generate a sequence of motion tokens matching the following human motion description. You can use the video as a reference. Video information: {image_token}\n Motion description: {qs}'
    return PROMPT
def get_prompt_text_only(qs):
    PROMPT = f'Generate a sequence of motion tokens matching the following human motion description.\n Motion description: {qs}'
    return PROMPT


def truncate_output_to_eos(output, eos_id):
    try:
        eos_pos = output.tolist().index(eos_id)
        output = output[:eos_pos+1]
    except ValueError:
        pass
    return output


def pad_left(x, max_len, pad_id):
    # pad right based on the longest sequence
    n = max_len - len(x)
    return torch.cat((torch.full((n,), pad_id, dtype=x.dtype).to(x.device), x))



@torch.no_grad()        
def vqvae_evaluation(out_dir, val_loader, net, logger, writer, eval_wrapper, nb_iter, best_fid=1000, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, is_train=False): 
    net.eval()
    nb_sample = 0

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


            pred_pose, loss_commit, perplexity = net(motion[i:i+1, :m_length[i]])
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())

            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose

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
    logger.info(msg)
    
    writer.add_scalar('./Test/FID', fid, nb_iter)
    writer.add_scalar('./Test/Diversity', diversity, nb_iter)
    writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
    writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
    writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
    writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)
    
    if is_train:
        if fid < best_fid : 
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            logger.info(msg)
            best_fid = fid
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

        if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
            logger.info(msg)
            best_div = diversity
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

        if R_precision[0] > best_top1 : 
            msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
            logger.info(msg)
            best_top1 = R_precision[0]
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

        if R_precision[1] > best_top2 : 
            msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
            logger.info(msg)
            best_top2 = R_precision[1]
        
        if R_precision[2] > best_top3 : 
            msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
            logger.info(msg)
            best_top3 = R_precision[2]
        
        if matching_score_pred < best_matching : 
            msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
            logger.info(msg)
            best_matching = matching_score_pred
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()        
def evaluation(params: EvaluationParams):
    stop_str = "<|end|>"
    conv_mode = "phi3_instruct"
    val_loader = params.val_loader
    net = params.net
    model = params.model
    logger = params.logger
    tokenizer = params.tokenizer
    tokenizer_image_token = params.tokenizer_image_token
    eval_wrapper = params.eval_wrapper
    temperature = params.temperature
    video_dir = params.video_dir
    image_processor = params.image_processor
    video_processor = params.video_processor
    start_id = params.start_id
    end_id = params.end_id
    out_dir = params.out_dir
    retrieval_result = params.retrieval_result
    text_only = params.text_only

    cache_keypoints = torch.load('../resources/motionbertprocessed.pth')
    if text_only:
        print('text only mode start !!!!')
    #. for retrieval settings
    if retrieval_result:
        with open(retrieval_result,'r') as f:
            top1_dict = json.load(f)

    model.eval()

    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for batch_id,batch in enumerate(tqdm(val_loader)):
        if start_id!=-1:

            if batch_id<start_id or batch_id>=end_id:
                    continue
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        # print(pose.mean())


        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
        pred_len = torch.ones(bs).long()

        encoded = []
 
        all_generated_text = []
        for k in range(bs):
            if text_only == False:
                new_name = clip_text[k]
                if retrieval_result:
                    # print('loading retrieval results')
                    video_path = top1_dict[new_name].replace('workspace','data/nas')
                if os.path.exists(video_path):
                    video_frames, context_frames = _get_rawvideo_dec(
                        video_path,
                        image_processor,
                        video_processor,
                        max_frames=NUM_FRAMES,
                        frame_resolution=224,
                        num_video_frames=NUM_FRAMES,
                        num_context_images=NUM_CONTEXT_IMAGES,
                    )
                    keypoints = _get_motion_motionbert(cache_keypoints,[video_path],16)
                else:
                    print('the video path does not exist... ',video_path)
                    exit()
                # Prepare query string with image tokens

                qs = get_prompt(DEFAULT_IMAGE_TOKEN * 16,clip_text[k])
            else:
                qs = get_prompt_text_only(clip_text[k])
            # Create conversation prompt
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
            with torch.inference_mode():
                if text_only == False:
                    output_ids = model.generate(
                        input_ids,
                        images=torch.stack(video_frames).half().cuda(), #[16,3,224,224]
                        context_images=torch.stack(context_frames).half().cuda(), #[16,3,336,336]
                        keypoints = keypoints,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        max_new_tokens=seq,
                        top_k=200,
                        use_cache=True,
                    )
                else:
                    output_ids = model.generate(
                        input_ids,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        max_new_tokens=seq,
                        top_k=200,
                        use_cache=True,
                    )
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs.replace("<|end|>", "")
            outputs = outputs.strip()  #'230,402,492'
            all_generated_text.append(outputs)


        for k in range(bs):
            all_tokens = []
            for i in all_generated_text[k].split(','):
                try:
                    cur_token = int(i)
                    if cur_token<=511:
                        all_tokens.append(cur_token)
                except:
                    continue
            
            try:
                assert len(all_tokens)!=0
                index_motion = torch.tensor(all_tokens).cuda()
            except:
                print('generating wrong format',all_generated_text[k])
                index_motion = torch.ones(1, seq).cuda().long()
                # exit()
            pred_pose = net.forward_decoder(index_motion) #index_motion:[1,49] tensor,   pred_pose [1,196,263]
            cur_len = pred_pose.shape[1]

            pred_len[k] = min(cur_len, seq)
            pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True) #array([14,20,24]). temp_match 94.46
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True) #array([9,16,21]) temp_match [120,82]
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs
    # print(nb_sample) #1504
    # exit()
    if start_id!=-1:
        print(f'start_id {str(start_id)} saved!')
        intermediate_data = {}
        intermediate_data['motion_annotation_list'] = motion_annotation_list
        intermediate_data['motion_pred_list'] = motion_pred_list
        intermediate_data['R_precision_real'] = R_precision_real
        intermediate_data['R_precision'] = R_precision
        intermediate_data['matching_score_real'] = matching_score_real
        intermediate_data['matching_score_pred'] = matching_score_pred
        intermediate_data['nb_sample'] = nb_sample
        torch.save(intermediate_data, os.path.join(out_dir,f'{str(start_id)}.pth'))
        return [None]*7

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

    msg = f"FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    model.train()
    return fid, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, logger




@torch.no_grad()        
def llm_inference_demo(params: EvaluationParams):
    stop_str = "<|end|>"
    conv_mode = "phi3_instruct"
    # val_loader = params.val_loader
    # net = params.net
    model = params.model
    # logger = params.logger
    tokenizer = params.tokenizer
    tokenizer_image_token = params.tokenizer_image_token
    # eval_wrapper = params.eval_wrapper
    temperature = params.temperature
    # video_dir = params.video_dir
    image_processor = params.image_processor
    video_processor = params.video_processor
    start_id = params.start_id
    end_id = params.end_id
    # out_dir = params.out_dir
    retrieval_result = params.retrieval_result
    text_only = params.text_only
    llm_generated = params.generated_file
    motion_encoder = params.motion_encoder
    if motion_encoder:
        cache_keypoints = torch.load('../resources/motionbertprocessed.pth')
    if text_only:
        print('text only mode start !!!!')
    #. for retrieval settings
    input_texts = []
    if retrieval_result:
        with open(retrieval_result,'r') as f:
            top1_dict = json.load(f)

        for key,value in top1_dict.items():
            input_texts.append(key)
    model.eval()
    # import ipdb;ipdb.set_trace()
    nb_sample = 0
    seq = 512
    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    final_dict = {}


    all_generated_text = []
    bs = len(input_texts)
    for k in range(bs):
        if text_only == False:
            # new_name = name[k]
            cur_text = input_texts[k]
            if retrieval_result:
                # print('loading retrieval results')
                video_path = top1_dict[cur_text].replace('workspace','data/nas')
            if os.path.exists(video_path):
                video_frames, context_frames = _get_rawvideo_dec(
                    video_path,
                    image_processor,
                    video_processor,
                    max_frames=NUM_FRAMES,
                    frame_resolution=224,
                    num_video_frames=NUM_FRAMES,
                    num_context_images=NUM_CONTEXT_IMAGES,
                )
                if motion_encoder:
                    keypoints = _get_motion_motionbert(cache_keypoints,[video_path],16)
                else:
                    keypoints = None
            else:
                print('the video path does not exist... ',video_path)
                exit()
            # Prepare query string with image tokens

            qs = get_prompt(DEFAULT_IMAGE_TOKEN * 16,input_texts[k])
        else:
            cur_text = input_texts[k]
            qs = get_prompt_text_only(clip_text[k])
            # qs = get_prompt_retrieved_motion_token(input_texts[k]) #for ablation study
        # Create conversation prompt
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        with torch.inference_mode():
            if text_only == False:
                # t0 = time.perf_counter()
                output_ids = model.generate(
                    input_ids,
                    images=torch.stack(video_frames).half().cuda(), #[16,3,224,224]
                    context_images=torch.stack(context_frames).half().cuda(), #[16,3,336,336]
                    keypoints = keypoints,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=seq,
                    top_k=200,
                    use_cache=True,
                )

            else:
                output_ids = model.generate(
                    input_ids,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=seq,
                    top_k=200,
                    use_cache=True,
                )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        # t = time.perf_counter() - t0
        # print(f"\n\nTime for inference: {t:.02f} sec total, {seq / t:.02f} tokens/sec", file=sys.stderr)
        # exit()
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        outputs = outputs.replace("<|end|>", "")
        outputs = outputs.strip()  #'230,402,492'
        all_generated_text.append(outputs)

        final_dict[cur_text] = outputs
        # final_dict.append(outputs)
    print(final_dict)
    with open(llm_generated,'w') as f:
        json.dump(final_dict,f)




@torch.no_grad()        
def llm_inference_func(params: EvaluationParams):
    stop_str = "<|end|>"
    conv_mode = "phi3_instruct"
    val_loader = params.val_loader
    # net = params.net
    model = params.model
    # logger = params.logger
    tokenizer = params.tokenizer
    tokenizer_image_token = params.tokenizer_image_token
    eval_wrapper = params.eval_wrapper
    temperature = params.temperature
    # video_dir = params.video_dir
    image_processor = params.image_processor
    video_processor = params.video_processor
    start_id = params.start_id
    end_id = params.end_id

    retrieval_result = params.retrieval_result
    text_only = params.text_only
    llm_generated = params.generated_file
    motion_encoder = params.motion_encoder
    if motion_encoder:
        cache_keypoints = torch.load('../resources/motionbertprocessed.pth')
    if text_only:
        print('text only mode start !!!!')
    #. for retrieval settings
    if retrieval_result:
        with open(retrieval_result,'r') as f:
            top1_dict = json.load(f)

    model.eval()
    # import ipdb;ipdb.set_trace()
    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    final_dict = {}
    # final_dict = []
    for batch_id,batch in enumerate(tqdm(val_loader)):
        if start_id!=-1:

            if batch_id<start_id or batch_id>=end_id:
                    continue
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        # print(pose.mean())


        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
        pred_len = torch.ones(bs).long()

        encoded = []
 
        all_generated_text = []
        for k in range(bs):

            if text_only == False:
                # new_name = name[k]
                cur_text = clip_text[k]
                if retrieval_result:

                    video_path = top1_dict[cur_text].replace('workspace','data/nas')
                if os.path.exists(video_path):
                    video_frames, context_frames = _get_rawvideo_dec(
                        video_path,
                        image_processor,
                        video_processor,
                        max_frames=NUM_FRAMES,
                        frame_resolution=224,
                        num_video_frames=NUM_FRAMES,
                        num_context_images=NUM_CONTEXT_IMAGES,
                    )
                    if motion_encoder:
                        keypoints = _get_motion_motionbert(cache_keypoints,[video_path],16)
                    else:
                        keypoints = None
                else:
                    print('the video path does not exist... ',video_path)
                    exit()
                # Prepare query string with image tokens

                qs = get_prompt(DEFAULT_IMAGE_TOKEN * 16,clip_text[k])
            else:
                cur_text = clip_text[k]
                qs = get_prompt_text_only(clip_text[k])
                # qs = get_prompt_retrieved_motion_token(clip_text[k])
            # Create conversation prompt
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
            with torch.inference_mode():
                if text_only == False:
                    # t0 = time.perf_counter()
                    output_ids = model.generate(
                        input_ids,
                        images=torch.stack(video_frames).half().cuda(), #[16,3,224,224]
                        context_images=torch.stack(context_frames).half().cuda(), #[16,3,336,336]
                        keypoints = keypoints,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        max_new_tokens=seq,
                        top_k=200,
                        use_cache=True,
                    )

                else:
                    output_ids = model.generate(
                        input_ids,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        max_new_tokens=seq,
                        top_k=200,
                        use_cache=True,
                    )
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
         
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs.replace("<|end|>", "")
            outputs = outputs.strip()  #'230,402,492'
            all_generated_text.append(outputs)
            final_dict[cur_text] = outputs


    with open(llm_generated,'w') as f:
        json.dump(final_dict,f)




@torch.no_grad()
def evaluation_for_generated_tokens(params: EvaluationParams):
    val_loader = params.val_loader
    net = params.net
    eval_wrapper = params.eval_wrapper

    start_id = params.start_id
    end_id = params.end_id
    out_dir = params.out_dir
    generated_file = params.generated_file
  

    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0

    with open(generated_file,'r') as f:
        all_generated_text = json.load(f)
    cur_index = 0

    for batch_id,batch in enumerate(tqdm(val_loader)):
        if start_id!=-1:

            if batch_id<start_id or batch_id>=end_id:
                    continue
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
       
        bs, seq = pose.shape[:2]
        # print(pose.mean())


        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
        pred_len = torch.ones(bs).long()

        encoded = []
 

        for k in range(bs):
            all_tokens = []
            for i in all_generated_text[clip_text[k]].split(','):
            # for i in all_generated_text[cur_index].split(','):
                try:
                    cur_token = int(i)
                    if cur_token<=511 and cur_token>=0:
                        all_tokens.append(cur_token)
                except:
                    continue
            
            # tokens = torch.tensor([int(token) for token in outputs.split(',')]).cuda()  motiongpt used
            try:
                assert len(all_tokens)!=0
                index_motion = torch.tensor(all_tokens).cuda()
            except:
                print('generating wrong format')
                print(all_generated_text[clip_text[k]])
                index_motion = torch.ones(1, seq).cuda().long()
                # exit()
            pred_pose = net.forward_decoder(index_motion) #index_motion:[1,49] tensor,   pred_pose [1,196,263]
            cur_len = pred_pose.shape[1]

            pred_len[k] = min(cur_len, seq)
            pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
            cur_index +=1

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
        # import ipdb;ipdb.set_trace()
        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)

        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True) #array([14,20,24]). temp_match 94.46
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True) #array([9,16,21]) temp_match [120,82]
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs
    # print(nb_sample) #1504
    # exit()
    if start_id!=-1:
        print(f'start_id {str(start_id)} saved!')
        intermediate_data = {}
        intermediate_data['motion_annotation_list'] = motion_annotation_list
        intermediate_data['motion_pred_list'] = motion_pred_list
        intermediate_data['R_precision_real'] = R_precision_real
        intermediate_data['R_precision'] = R_precision
        intermediate_data['matching_score_real'] = matching_score_real
        intermediate_data['matching_score_pred'] = matching_score_pred
        intermediate_data['nb_sample'] = nb_sample
        torch.save(intermediate_data, os.path.join(out_dir,f'{str(start_id)}.pth'))
        return [None]*7
    # import ipdb;ipdb.set_trace()
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy() #[1504,512]
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np) #(512,) cov (512,512)
    mu, cov= calculate_activation_statistics(motion_pred_np)
    # import ipdb;ipdb.set_trace()
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    # logger.info(msg)
    print(msg)

    # model.train()
    return fid, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred



@torch.no_grad()
def generate_preference_data(params: EvaluationParams):
    train_loader = params.val_loader
    net = params.net
    train_wrapper = params.eval_wrapper
    generated_file = params.generated_file
    candidate_files = params.candidate_files
    fid_weight = params.fid_weight
    match_weight = params.match_weight

    all_candidates = {} #key: input_text, value [cand1,cand2,cand3,...]
    print('loading the dataset...')
    for file in tqdm(candidate_files):
        with open(file,'r') as f:
            generated_text = json.load(f)
        for key,value in generated_text.items():
            if key in all_candidates:
                all_candidates[key].append(value)
            else:
                all_candidates[key] = [value]
    final_result = {}
    final_result_chosen = {}
    final_result_rejected = {}

    for batch_id,batch in enumerate(tqdm(train_loader)):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        try:
            pose = pose.cuda().float()
        except:
            continue
            # import ipdb;ipdb.set_trace()
        # exit()
        bs, seq = pose.shape[:2]

        for k in range(bs):
            try:
                cur_generated_motions = all_candidates[clip_text[k]]
            except:
                continue
            cur_candi_fid = []
            cur_candi_match = []
            try:
                et, em = train_wrapper.get_co_embeddings(word_embeddings[k:k+1,:,:], pos_one_hots[k:k+1,:,:], sent_len[k:k+1], pose[k:k+1,:,:], m_length[k:k+1])
            except:
                
                continue
            motion_annotation_list = [em]
            motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
            gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
            for pre_text in cur_generated_motions:
                pred_pose_eval = torch.zeros((1, seq, pose.shape[-1])).cuda()
                pred_len = torch.ones(1).long()

                all_tokens = []
                for i in pre_text.split(','):
                    try:
                        cur_token = int(i)
                        if cur_token<=511 and cur_token>=0:
                            all_tokens.append(cur_token)
                    except:
                        continue
            
                try:
                    assert len(all_tokens)!=0
                    index_motion = torch.tensor(all_tokens).cuda()
                except:
                    print('generating wrong format',pre_text)
                    index_motion = torch.ones(1, seq).cuda().long()
                    # exit()
                pred_pose = net.forward_decoder(index_motion) #index_motion:[1,49] tensor,   pred_pose [1,196,263]
                cur_len = pred_pose.shape[1]

                pred_len[0] = min(cur_len, seq)
                pred_pose_eval[:1, :cur_len] = pred_pose[:, :seq] #TODO:
                try:
                    et_pred, em_pred = train_wrapper.get_co_embeddings(word_embeddings[k:k+1,:,:], pos_one_hots[k:k+1,:,:], sent_len[k:k+1], pred_pose_eval, pred_len)
                except:
                    
                    continue
                motion_pred_list = [em_pred]
                
                motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
                
                mu, cov= calculate_activation_statistics(motion_pred_np)
                fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

                temp_R, match_score = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=1, sum_all=True) #array([14,20,24]). temp_match 94.46
                cur_candi_fid.append(fid)
                cur_candi_match.append(match_score)
            norm_fid_scores = softmax(np.array(cur_candi_fid))
            norm_match_scores = softmax(np.array(cur_candi_match))
            overall_score = []
            for item_index,cur_fid in enumerate(norm_fid_scores):
                overall_score.append(fid_weight*cur_fid+match_weight*norm_match_scores[item_index])
            min_index = overall_score.index(min(overall_score))
            max_index = overall_score.index(max(overall_score))
            temp_dict = {}
            temp_dict['chosen'] = cur_generated_motions[min_index]
            temp_dict['rejected'] = cur_generated_motions[max_index]
            final_result[clip_text[k]] = temp_dict
            final_result_chosen[clip_text[k]] = cur_generated_motions[min_index]
            final_result_rejected[clip_text[k]] = cur_generated_motions[max_index]

    with open(generated_file,'w') as f:
        json.dump(final_result,f)
    with open(generated_file.replace('.json','_chosen.json'),'w') as f:
        json.dump(final_result_chosen,f)
    with open(generated_file.replace('.json','_rejected.json'),'w') as f:
        json.dump(final_result_rejected,f)




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


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
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
