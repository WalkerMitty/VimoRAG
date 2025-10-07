
from mcdpo.model.internvideo_1.modeling import CLIP4Clip
import argparse
import torch
from dataclasses import dataclass, field
from mcdpo.model.internvideo_1.modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import os
import numpy as np
from torch import nn
import cv2

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def normalize(data):
    return (data / 255.0 - v_mean) / v_std


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert (len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube


def get_vid_feat(frames, vlm):
    return vlm.get_vid_features(frames)


def retrieve_vision(frames, model, topk: int = 5, config: dict = {}, device=torch.device('cuda')):
    vlm = model
    vlm = vlm.to(device)

    fn = config.get('num_frames', 8)
    size_t = config.get('size_t', 224)
    frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t), device=device)
    vision_embeds, pooled_vision_embeds = vlm.get_vid_feat(frames_tensor)

    return vision_embeds, pooled_vision_embeds

@dataclass
class Args:
    video_dim: int = 1024
    seed: int = 42
    max_words: int = 77
    max_frames: int = 12
    feature_framerate: int = 1
    margin: float = 0.1
    hard_negative_rate: float = 0.5
    negative_weighting: int = 1
    n_pair: int = 1
    cross_model: str = 'cross-base'
    init_model: str = None
    resume_model: str = None
    resume_model_true: str = None
    do_lower_case: bool = False
    warmup_proportion: float = 0.1
    gradient_accumulation_steps: int = 1
    n_gpu: int = 1
    cache_dir: str = ''
    fp16: bool = False
    fp16_opt_level: str = 'O1'
    task_type: str = 'retrieval'
    datatype: str = 'msrvtt'
    world_size: int = 0
    local_rank: int = 0
    rank: int = 0
    coef_lr: float = 1.0
    use_mil: bool = False
    sampled_use_mil: bool = False
    text_num_hidden_layers: int = 12
    visual_num_hidden_layers: int = 12
    cross_num_hidden_layers: int = 4
    loose_type: bool = True
    expand_msrvtt_sentences: bool = False
    train_frame_order: int = 0
    eval_frame_order: int = 0
    freeze_layer_num: int = 0
    slice_framepos: int = 2
    linear_patch: str = '2d'
    sim_header: str = 'meanP'
    mlp_layer: int = 1
    mlp_lr: float = 1e-4
    stage: int = 1
    weight_fc_path: str = ''
    pretrained_clip_name: str = 'ViT-L/14'
    pretrained_path: str = '../resources/InternVideo-MM-L-14.ckpt'
    clip_evl: bool = True
    mergeclip: bool = False
    mergeweight: float = 0.5
    interaction: str = 'no'
    wti_arch: int = 0
    cdcr: int = 0



def build_internvideo_1(model_path):

    intern_model = setup_internvideo1V(model_path)
    return intern_model

def setup_internvideo1V(model_path):
    model = InternVideo1V(model_path)
    model = model.to(torch.device('cuda'))
    model_without_ddp = model
    model_without_ddp = model_without_ddp.to(torch.float32)

    model_without_ddp.eval()
    return (model_without_ddp)



class InternVideo1V(nn.Module):
    """docstring for InternVideo1V"""

    def __init__(self,model_path):
        super(InternVideo1V, self).__init__()

        # create modules.
        self.vision_encoder = self.build_vision_encoder(model_path)
        # import ipdb;ipdb.set_trace()
        self.freeze_vision()
        self.vision_width = 768
        self.embed_dim = 512
        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)

        # self.image_processor = VideoTrainProcessor(num_frames=self.vision_encoder.num_frames)
        self.image_processor = VideoTrainProcessor()
    def load_model(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        msg = self.load_state_dict(checkpoint, strict=False)
        # print(f"load_state_dict: {msg}")

    def freeze_vision(self):
        """freeze vision encoder"""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    @property
    def dtype(self):
        # return self.vision_encoder.patch_embed.proj.weight.dtype
        return self.vision_encoder.clip.token_embedding.weight.dtype

    def encode_vision(self, image: torch.Tensor, test: bool = False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
            - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """

        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        # keep_temporal=self.config.model.vision_encoder.keep_temporal
        if test:
            vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(
                image, None, use_image
            )
            return vision_embeds, pooled_vision_embeds
        else:
            mask, targets_clip_middle_vis, targets_clip_final_vis = self.encode_teacher(image)
            # if mask is not None and (self.video_mask_type != 'tube' or self.image_mask_type != 'tube'):
            #     keep_temporal = False
            # print(f"\033[31mmask is {type(mask)}\033[0m")
            vision_embeds, pooled_vision_embeds, student_output, student_output_final = self.vision_encoder(
                image, mask, use_image
            )
            return vision_embeds, pooled_vision_embeds, student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis

    def forward(self, image):
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B,T,C,H,W] -> [B,C,T,H,W]

        self.vision_encoder.clip.visual.t_size = 4 #frame number
        self.vision_encoder.clip.visual.transformer.T = 4
        # import ipdb;ipdb.set_trace()
        if use_image:
            mode = 'image'
        else:
            mode = 'video'
        evl_output, visual_output = self.vision_encoder.clip.encode_video(image, mode=mode,return_all_feats=True)
        # evl_output [bsz, 768]
        # visual_output [bsz, frame_number, 768]
        # import ipdb;ipdb.set_trace()


        # return vision_embeds
        return visual_output

    def build_vision_encoder(self,model_path):
        """build vision encoder
        Returns: (vision_encoder, clip_teacher). Each is a `nn.Module`.

        """

        args = Args()
        model_state_dict = torch.load(model_path, map_location='cpu')
        # cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        vision_encoder = CLIP4Clip.from_pretrained(args.cross_model, cache_dir='', state_dict=model_state_dict, task_config=args)
        # # parameters for mask


        return vision_encoder

    def get_vid_feat(self, frames: torch.Tensor):
        """get the video features for the given frames.

        Args:
            frames (torch.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].

        """
        with torch.no_grad():
            vision_embeds, pooled_vision_embeds = self.encode_vision(
                frames, test=True
            )  # vfeat = self.vision_proj(vfeat)  # vfeat /= vfeat.norm(dim=-1, keepdim=True)
        return vision_embeds, pooled_vision_embeds

    @property
    def hidden_size(self):
        return self.vision_encoder.embed_dim

    @property
    def num_patches(self):
        return self.config.model.vision_encoder.patch_size


class VideoTrainProcessor():
    def __init__(self, image_size=(224, 224), mean=None, std=None, num_frames=8):
        super().__init__()

        if mean is None:
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        if std is None:
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        self.mean = mean
        self.std = std
        self.num_frames = num_frames

    def normalize(self, data):
        return (data / 255.0 - self.mean) / self.std

    def frames2tensor(self, vid_list, target_size=(224, 224), use_image=False):
        # Ensure we have at least `self.num_frames`
        if not use_image:
            assert (len(vid_list) >= self.num_frames)

        # Process each frame
        vid_list = [cv2.resize(x, target_size) for x in vid_list]
        vid_tube = [normalize(x) for x in vid_list]
        vid_tube = [np.transpose(x, (2, 0, 1)) for x in vid_tube]
        vid_tube = [torch.from_numpy(x) for x in vid_tube]

        return vid_tube

    def preprocess(self, vid_list, use_image=False):
        return {'pixel_values': self.frames2tensor(vid_list, use_image=use_image)}