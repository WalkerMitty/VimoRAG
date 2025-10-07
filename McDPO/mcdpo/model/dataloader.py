from decord import VideoReader, cpu
from mmengine import fileio
import io
import numpy as np
import torch


# import mcdpo.model.wham_dataloader.constants as _C


# from mcdpo.model.wham_dataloader import transforms
# from mcdpo.model.wham_dataloader.kp_utils import root_centering
# from mcdpo.model.wham_dataloader.imutils import compute_cam_intrinsics
from mcdpo.model.internvideo_1.modules import clip_evl
from mcdpo.model.internvideo_1.modules.module_clip import CLIP
import os,sys





def uniform_sample(lst, n):
    assert n <= len(lst)
    m = len(lst)
    step = m // n  # Calculate the step size
    return [lst[i * step] for i in range(n)]

def _get_rawvideo_dec(video_path, image_processor, video_processor, max_frames=8, frame_resolution=224,
                      video_framerate=1, s=None, e=None, min_frames=8, num_video_frames=8, num_context_images=8):

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    try:
        vreader = VideoReader(video_path, num_threads=1)
    except Exception as e:
        video_bytes = fileio.get(video_path)
        vreader = VideoReader(io.BytesIO(video_bytes), num_threads=1)

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        elif len(all_pos) < min_frames:
            if num_frames < min_frames:
                min_frames = num_frames
            t_stride = max(1, (f_end - f_start) // (min_frames - 1))
            adjusted_f_end = f_start + t_stride * (min_frames - 1)
            sample_pos = list(range(f_start, adjusted_f_end + 1, t_stride))
        else:
            sample_pos = all_pos

        all_images = [f for f in vreader.get_batch(sample_pos).asnumpy()] # a list of numpy [1920, 1080,3]
        # In case if we can't sample MAX_IMAGE_LENGTH frames
        num_video_frames_sampled = min(num_video_frames, len(all_images))
        num_context_images_sampled = min(num_context_images, len(all_images))

        video_frames = uniform_sample(all_images, num_video_frames_sampled) # a list of numpy [1920, 1080,3]
        context_images = uniform_sample(all_images, num_context_images_sampled)

        video_frames = video_processor.preprocess(video_frames)['pixel_values']
        context_images = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in context_images]

        if len(context_images) < num_context_images:  # Pad
            while len(context_images) < num_context_images:
                context_images.append(
                    torch.zeros((3, image_processor.crop_size['height'], image_processor.crop_size['width'])))

        slice_len = len(video_frames)

        if slice_len < 1:
            pass
        else:
            #mask: tensor [8,17]
            # padding_num = 0
            while len(video_frames) < num_video_frames:
                video_frames.append(torch.zeros((3, frame_resolution, frame_resolution)))
                # mask.append(torch.zeros)

    else:
        print("video path: {} error.".format(video_path))
    # import ipdb;ipdb.set_trace()
    return video_frames, context_images





# def _obtain_keypoints(video_path, max_frames=8, frame_resolution=224,
#                       video_framerate=1, s=None, e=None, min_frames=8, num_video_frames=8, num_context_images=8,
#                       detector=None,extractor=None,smpl=None,keypoints_normalizer=None):
#     to = lambda x: x.unsqueeze(0)
#     detector.initialize_tracking()
#     if s is None:
#         start_time, end_time = None, None
#     else:
#         start_time = int(s)
#         end_time = int(e)
#         start_time = start_time if start_time >= 0. else 0.
#         end_time = end_time if end_time >= 0. else 0.
#         if start_time > end_time:
#             start_time, end_time = end_time, start_time
#         elif start_time == end_time:
#             end_time = start_time + 1

#     try:
#         vreader = VideoReader(video_path, num_threads=1)
#     except Exception as e:
#         video_bytes = fileio.get(video_path)
#         vreader = VideoReader(io.BytesIO(video_bytes), num_threads=1)

#     fps = vreader.get_avg_fps()
#     f_start = 0 if start_time is None else int(start_time * fps)
#     f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
#     num_frames = f_end - f_start + 1
#     device = next(smpl.parameters()).device
#     if num_frames > 0:
#         # T x 3 x H x W
#         sample_fps = int(video_framerate)
#         t_stride = int(round(float(fps) / sample_fps))

#         all_pos = list(range(f_start, f_end + 1, t_stride))
#         if len(all_pos) > max_frames:
#             sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
#         elif len(all_pos) < min_frames:
#             if num_frames < min_frames:
#                 min_frames = num_frames
#             t_stride = max(1, (f_end - f_start) // (min_frames - 1))
#             adjusted_f_end = f_start + t_stride * (min_frames - 1)
#             sample_pos = list(range(f_start, adjusted_f_end + 1, t_stride))
#         else:
#             sample_pos = all_pos

#         all_images = [f for f in vreader.get_batch(sample_pos).asnumpy()] # a list of numpy [1920, 1080,3]
#         # In case if we can't sample MAX_IMAGE_LENGTH frames
#         num_video_frames_sampled = min(num_video_frames, len(all_images))
#         num_context_images_sampled = min(num_context_images, len(all_images))

#         video_frames = uniform_sample(all_images, num_video_frames_sampled) # a list of numpy [1920, 1080,3]
#         context_images = uniform_sample(all_images, num_context_images_sampled)
        
        
#         #<<<---------
#         # here obtain the 2d key points
#         width = video_frames[0].shape[1]
#         height = video_frames[0].shape[0]
#         for img in video_frames:
#             detector.track(img, fps,0) #here 0 is not used
       
#         try:
#             tracking_results = detector.process(fps)
#             tracking_results = extractor.run(video_path, tracking_results)
#             # import ipdb;ipdb.set_trace()
#             # Process 2D keypoints
#             prefix = 'flipped_'
#             kp2d = torch.from_numpy(tracking_results[0][prefix+'keypoints']).float()
#             mask = kp2d[..., -1] < 0.3
#             bbox = torch.from_numpy(tracking_results[0][prefix + 'bbox']).float()
            
#             res = torch.tensor([width,height]).float()
#             intrinsics = compute_cam_intrinsics(res)
#             norm_kp2d, _ = keypoints_normalizer(
#                 kp2d[..., :-1].clone(), res, intrinsics, 224, 224, bbox
#             )
#             # Process initial pose
#             # import ipdb;ipdb.set_trace()
            
#             init_output = smpl.get_output(
#                 global_orient=tracking_results[0][prefix + 'init_global_orient'].to(device),
#                 body_pose=tracking_results[0][prefix + 'init_body_pose'].to(device),
#                 betas=tracking_results[0][prefix + 'init_betas'].to(device),
#                 pose2rot=False,
#                 return_full_pose=True
#             )
#             # import ipdb;ipdb.set_trace()
#             init_kp3d = root_centering(init_output.joints[:, :17].cpu(), 'coco')
#             init_kp = torch.cat((init_kp3d.reshape(1, -1), norm_kp2d[0].clone().reshape(1, -1)), dim=-1)
#             init_smpl = transforms.matrix_to_rotation_6d(init_output.full_pose.cpu())
#             # print('detected something!')
#         except:
#             #does not detect anything
#             # print('does not detect any thing!')
#             mask = torch.zeros(1, 17, dtype=torch.bool)
#             norm_kp2d = torch.zeros(1, 37,dtype=next(smpl.parameters()).dtype)
#             init_kp = torch.zeros(1, 88,dtype=norm_kp2d.dtype)
#             init_smpl = torch.zeros(1, 24,6,dtype=norm_kp2d.dtype)


#         slice_len = len(video_frames)

#         if slice_len < 1:
#             pass
#         else:
#             #mask: tensor [8,17]
#             # padding_num = 0

#                 # mask.append(torch.zeros)
#             padding_num = len(video_frames) - mask.size()[0]
#             if padding_num!=0:
#                 temp = torch.zeros(padding_num, 17, dtype=torch.bool)
#                 temp2 = torch.zeros(padding_num, 37,dtype=norm_kp2d.dtype)
#                 # import ipdb;ipdb.set_trace()
#                 mask = torch.cat((mask,temp),dim=0)
#                 norm_kp2d = torch.cat((norm_kp2d,temp2),dim=0)



#     else:
#         print("video path: {} error.".format(video_path))
#     import ipdb;ipdb.set_trace()
#     return to(norm_kp2d),to(init_kp),to(init_smpl),to(mask)
def build_clip_model():
    clip_state_dict = CLIP.get_config(pretrained_clip_name='ViT-L/14')
    clip_state_dict = clip_state_dict['state_dict'] if 'state_dict' in clip_state_dict else clip_state_dict
    pretrained_path = '../resources/InternVideo-MM-L-14.ckpt'
    clip, _ = clip_evl.load(pretrained_path, t_size=12, mergeclip=False, mergeweight=0.5, clip_state_dict=clip_state_dict)
    return clip

def _get_fixed_text_fea(text,clip,tokenizer):
    words = tokenizer.tokenize(text)
    SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>","MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    max_words = 77 
    total_length_with_CLS = max_words - 1
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < max_words:
        input_ids.append(0)
    assert len(input_ids) == max_words
    input_ids = np.array(input_ids)
    # import ipdb;ipdb.set_trace()
    device = next(clip.parameters()).device
    input_ids = torch.from_numpy(input_ids).unsqueeze(0).to(device) #[1,77]
    return clip.encode_text(input_ids).float() #[1,768]

