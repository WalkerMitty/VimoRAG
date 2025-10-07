import cv2
import io
import imageio
import torch
import numpy as np
from decord import VideoReader
from PIL import Image
from mcdpo.constants import *
from mmengine import fileio
from mmengine.fileio import FileClient

client = FileClient('disk')

def uniform_sample(lst, n):
    assert n <= len(lst)
    m = len(lst)
    step = m // n                           
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

        all_images = [f for f in vreader.get_batch(sample_pos).asnumpy()]                                 

        num_video_frames_sampled = min(num_video_frames, len(all_images))
        num_context_images_sampled = min(num_context_images, len(all_images))

        video_frames = uniform_sample(all_images, num_video_frames_sampled)                                 
        context_images = uniform_sample(all_images, num_context_images_sampled)

        video_frames = video_processor.preprocess(video_frames)['pixel_values']
        context_images = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in context_images]

        if len(context_images) < num_context_images:       
            while len(context_images) < num_context_images:
                context_images.append(
                    torch.zeros((3, image_processor.crop_size['height'], image_processor.crop_size['width'])))

        slice_len = len(video_frames)

        if slice_len < 1:
            pass
        else:

            while len(video_frames) < num_video_frames:
                video_frames.append(torch.zeros((3, frame_resolution, frame_resolution)))

    else:
        print("video path: {} error.".format(video_path))

    return video_frames, context_images

def read_gif_mod(video_path, image_processor, max_frames=16, image_resolution=224, video_framerate=25,
                 s=None, e=None, sample_fps=1):

    video = np.zeros((max_frames, 3, image_resolution, image_resolution), dtype=np.float64)

    video_bytes = client.get(video_path)
    gif_reader = imageio.get_reader(io.BytesIO(video_bytes))
    num_frames = len(gif_reader)

    fps = video_framerate
    f_start = 0 if s is None else max(int(s * fps), 0)
    f_end = min(num_frames - 1, int(e * fps)) if e is not None else num_frames - 1

    t_stride = max(int(round(float(fps) / sample_fps)), 1)
    frame_indices = range(f_start, f_end + 1, t_stride)

    processed_frames = []
    for i, frame in enumerate(gif_reader):
        if i in frame_indices:
            img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            img_pil = Image.fromarray(img).resize((image_resolution, image_resolution))
            processed_frames.append(img_pil)

            if len(processed_frames) >= max_frames:
                break

    patch_images = processed_frames
    patch_images = image_processor.preprocess(patch_images)['pixel_values']

    slice_len = patch_images.shape[0]

    video[:slice_len, ...] = patch_images

    return video, slice_len

def read_frame_mod(video_path, image_processor, video_processor, max_frames=16, image_resolution=224, video_framerate=3,
                   s=None, e=None, sample_fps=1, num_video_frames=16, num_context_images=16):

    video = np.zeros((max_frames, 3, image_resolution, image_resolution), dtype=np.float64)
    max_video_length = 0

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path {video_path} not found.")

    frame_files = sorted(os.listdir(video_path))
    num_frames = len(frame_files)

    fps = video_framerate
    f_start = 0 if s is None else max(int(s * fps), 0)
    f_end = min(num_frames - 1, int(e * fps)) if e is not None else num_frames - 1

    t_stride = max(int(round(float(fps) / sample_fps)), 1)
    frame_indices = range(f_start, f_end + 1, t_stride)

    all_frames = []
    for idx in frame_indices:
        img_path = os.path.join(video_path, frame_files[idx])
        img = np.array(Image.open(img_path))
        all_frames.append(img)

        if len(all_frames) >= max_frames:
            break

    num_video_frames_sampled = min(num_video_frames, len(all_frames))
    num_context_images_sampled = min(num_context_images, len(all_frames))

    patch_images = uniform_sample(all_frames, num_video_frames_sampled)
    context_images = uniform_sample(all_frames, num_context_images_sampled)

    patch_images = video_processor.preprocess(patch_images)['pixel_values']
    context_images = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in context_images]

    if len(context_images) < num_context_images:       
        while len(context_images) < num_context_images:
            context_images.append(
                torch.zeros((3, image_processor.crop_size['height'], image_processor.crop_size['width'])))

    slice_len = len(patch_images)

    if slice_len < 1:
        pass
    else:
        while len(patch_images) < num_video_frames:
            patch_images.append(torch.zeros((3, image_resolution, image_resolution)))

    return patch_images, context_images, slice_len
