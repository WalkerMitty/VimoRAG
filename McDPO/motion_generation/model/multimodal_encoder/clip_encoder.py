import torch
import torch.nn as nn
import math
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args=None, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        if args is None:
            self.select_layer = -2
            self.select_feature = 'patch'
        else:
            self.select_layer = args.mm_vision_select_layer
            self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        # print('line 20 in clip_encoder')
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        # print('line 25 in clip encoder')
    def load_model(self):
        # print('line 27')
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        # print('line 29')
        self.image_eval_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        # print('line 31')
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs, select_feature='patch'):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, select_feature='patch', batch_size=128):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out, select_feature).to(image.dtype)
                image_features.append(image_feature)
        else:
            if images.shape[0] > batch_size:
                n_chunk = images.shape[0]
                image_features = []
                n_iter = int(math.ceil(n_chunk / float(batch_size)))
                for i in range(n_iter):
                    min_ind = i * batch_size
                    max_ind = (i + 1) * batch_size
                    batch = images[min_ind:max_ind]
                    batch_forward_outs = self.vision_tower(batch.to(device=self.device, dtype=self.dtype),
                                                           output_hidden_states=True)
                    batch_features = self.feature_select(batch_forward_outs, select_feature).to(batch.dtype)
                    image_features.append(batch_features)
                image_features = torch.cat(image_features)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                       output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs, select_feature).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
