from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn
import torch.nn.functional as F

from modules.until_module import PreTrainedModel, AllGather, CrossEn, LossWeight
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip
from modules.our_module_cross import Transformer as TransformerVision
from modules.xpool_transformer import Transformer as XpoolTransformer
from modules.xpool_transformer import Transformer_self
from modules.module_clip import CLIP, convert_weights
from modules import clip_evl
from modules.clip_evl.model_no_freeze_only_global import vit_only_global_l_sparse8_k400, vit_only_global_b_sparse8_k400
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

logger = logging.getLogger(__name__)
allgather = AllGather.apply
from einops import rearrange

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name

        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)                   

        clip_state_dict = clip_state_dict['state_dict'] if 'state_dict' in clip_state_dict else clip_state_dict
        for key, val in clip_state_dict.items():
            if key not in state_dict:
                state_dict[key] = val.clone()
            new_key = key.replace('clip.', '')
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        elif model.sim_header == "seqLSTM" or model.sim_header == "seqTransf" or model.sim_header =="xseq":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue

        elif model.sim_header == 'videoTransVision' or model.sim_header =='textTransVision' or model.sim_header == 'clsTransVision':
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerVision.")] = val.clone()
                            continue

        if state_dict is not None and task_config.clip_evl is False:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class dual_softmax_loss(nn.Module):
    def __init__(self,):
        super(dual_softmax_loss, self).__init__()

    def forward(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix/temp, dim=0)*len(sim_matrix)                                                                                  
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        if 'clip.visual.proj' in clip_state_dict:
            new_dict = {}
            for k, v in clip_state_dict.items():
                new_k = k.replace('clip.', '')
                new_dict[new_k] = v.clone()

            clip_state_dict = new_dict

        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]      
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]    
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)    
            image_resolution = vision_patch_size * grid_size     
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]     
        context_length = clip_state_dict["positional_embedding"].shape[0]    
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]       
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))

        self.interaction = task_config.interaction if hasattr(task_config, 'interaction') else 'no'
        self.wti_arch = task_config.wti_arch if hasattr(task_config, 'wti_arch') else 0
        self.mlp_layer = task_config.mlp_layer if hasattr(task_config, 'mlp_layer') else 0
        self.cdcr = task_config.cdcr if hasattr(task_config, 'cdcr') else 0

        if hasattr(task_config, "clip_evl") and task_config.clip_evl == True:                           
            self.clip, _ = clip_evl.load(task_config.pretrained_path, t_size=task_config.max_frames, mergeclip=task_config.mergeclip, mergeweight=task_config.mergeweight, clip_state_dict=clip_state_dict)
            self.clip = self.clip.float()         
            self.clip_evl = True

        else:
            self.clip_evl = False
            self.clip = CLIP(
                embed_dim,
                image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
                context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
                linear_patch=self.linear_patch,sim_header = task_config.sim_header, stage=task_config.stage
            ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        if not self.clip_evl:
            convert_weights(self.clip)                               

        self.stage = task_config.stage
        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:

            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)

            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)
        if self.sim_header == 'xseq':
            self.transformer = XpoolTransformer(768,1,0.3)
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, 768)
            self.transformer_self = Transformer_self(768,1)

        if self.sim_header == "xpool":
            self.transformer = XpoolTransformer(768,1,0.3)

        if self.sim_header in ["seqLSTM","videoTransVision",'clsTransVision','seqTransf']:

            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, 768)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )

        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)
        if self.sim_header == "videoTransVision" or self.sim_header == "textTransVision" or self.sim_header=='clsTransVision':
            self.transformerVision = TransformerVision(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, ) 

        if self.sim_header == "weightPrior":
            self.loss_weight = LossWeight()
            if self.mlp_layer == 1:

                self.video_weight_fc = nn.Linear(transformer_width, 1)

            elif self.mlp_layer == 2:

                self.video_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
            elif self.mlp_layer == 3:

                self.video_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
            elif self.mlp_layer == 4:
                self.video_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
            if self.stage==2:
                checkpoint = torch.load(task_config.weight_fc_path)

                self.video_weight_fc.load_state_dict(checkpoint)

        if self.sim_header == "biaffine":
            self.U1 = nn.Parameter(torch.randn(768, 768))
            self.U2 = nn.Parameter(torch.randn(768,1))

        if self.cdcr:
            self.cdcr_alpha1 = 0.16
            self.cdcr_alpha2 = 0
            self.cdcr_lambda = 0.001
        if self.interaction == 'wti' or self.interaction == 'ti':
            if self.wti_arch == 1:
                self.text_weight_fc = nn.Linear(transformer_width, 1)
                self.video_weight_fc = nn.Linear(transformer_width, 1)

            elif self.wti_arch == 2:
                self.text_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
                self.video_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
            elif self.wti_arch == 3:
                self.text_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
                self.video_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))

        self.loss_fct = CrossEn()

        if not self.clip_evl:
            self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        sequence_output, text_fea,visual_output,video_fea = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True, video_frame=video_frame)

        if self.training:
            loss = 0.
            cdcr_loss = 0.
            if self.wti_interaction != 'no' and self.cdcr:
                sim_matrix, _, cdcr_loss, *_ = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                    shaped=True, loose_type=self.loose_type)
            else:
                sim_matrix, *_tmp, self_video_sim = self.get_similarity_logits(sequence_output, attention_mask, visual_output,video_mask,video_fea,text_fea,
                                                   shaped=True, loose_type=self.loose_type)

            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2

            if self.sim_header=='weightPrior' and self.stage==1:
                weight_loss = self.loss_weight(self_video_sim)
                loss += weight_loss
                sim_loss = sim_loss * 0.
            loss += sim_loss

            if self.cdcr:
                loss += self.cdcr_lambda * cdcr_loss
            return loss
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)

        if self.interaction == 'wti' or self.interaction == 'ti':
            if self.clip_evl:
                sequence_hidden = self.clip.encode_text(input_ids, return_all_feats=True)[1].float()
            else:
                sequence_hidden = self.clip.encode_text(input_ids, return_hidden=True)[1].float()
        else:            

            sequence_hidden,text_fea = self.clip.encode_text(input_ids,return_all_feats=True)
            sequence_hidden = sequence_hidden.float()
            text_fea = text_fea.float()

        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden,text_fea

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        if self.clip_evl:
            if len(video.size()) == 4:

                video = video.view(bs_pair, -1, video.size(-3), video.size(-2), video.size(-1))
                video = video.permute(0, 2, 1, 3, 4).contiguous()

            visual_output,video_fea = self.clip.encode_video(video, return_all_feats=True)

            visual_output = visual_output.float()
            video_fea = video_fea.float()
            if self.interaction == 'wti':

                visual_hidden = self.clip.visual_ln_post(visual_output[0]) @ self.clip.visual_proj
        else:
            visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()

        return visual_output,video_fea

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
        sequence_output,text_fea = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output,video_fea = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)     

        return sequence_output, text_fea,visual_output,video_fea

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)                                
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)          
        visual_output = visual_output * video_mask_un                
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)         
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output,attention_mask, visual_output,video_mask, frames_fea,text_fea,sim_header="meanP"):

        visual_output = visual_output.unsqueeze(1)
        if sim_header == "meanP" or sim_header =='biaffine':

            pass
        elif sim_header == "seqLSTM":

            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":

            frames_fea_original = frames_fea.clone()
            seq_length = frames_fea.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=frames_fea.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            frames_fea = frames_fea + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            frames_fea = frames_fea.permute(1, 0, 2)              
            visual_output = self.transformerClip(frames_fea, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)              
            visual_output = visual_output + frames_fea_original

        elif sim_header == 'textTransVision':
            bsz = frames_fea.size()[0]
            expand_text_fea = text_fea.unsqueeze(1).repeat(1, bsz, 1, 1).view(-1, text_fea.size()[1], text_fea.size()[2])                                  
            expand_frames_fea = frames_fea.repeat(bsz, 1, 1)
            video_mask = video_mask.repeat(bsz,1)                        
            expand_att_mask = attention_mask.unsqueeze(1).repeat(1, bsz, 1).view(-1, attention_mask.size()[1])               

            expand_frames_fea = expand_frames_fea.permute(1,0,2)
            expand_text_fea = expand_text_fea.permute(1,0,2)
            visual_output = self.transformerVision(expand_text_fea, expand_frames_fea,expand_frames_fea,video_mask.bool()) 
            visual_output = visual_output.permute(1, 0, 2)                

        elif sim_header == 'xseq':
            frames_fea_original = frames_fea.clone()
            seq_length = frames_fea.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=frames_fea.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)

            frames_fea = frames_fea + frame_position_embeddings

            frames_fea = frames_fea.permute(1, 0, 2)              
            visual_output = self.transformer_self(frames_fea)
            visual_output = visual_output.permute(1, 0, 2)              
            frames_fea = visual_output + frames_fea_original
            output = self.transformer(sequence_output.squeeze(1),frames_fea)
            t2v = output.permute(1,0,2)
            bsz = frames_fea.size()[0]
            visual_output = t2v.reshape(bsz*bsz,-1)
        elif sim_header == 'xpool':

            output = self.transformer(sequence_output.squeeze(1),frames_fea)
            t2v = output.permute(1,0,2)
            bsz = frames_fea.size()[0]
            visual_output = t2v.reshape(bsz*bsz,-1)
        elif sim_header == 'clsTransVision':
            seq_length = frames_fea.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=frames_fea.device)
            position_ids = position_ids.unsqueeze(0).expand(frames_fea.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            frames_fea = frames_fea + frame_position_embeddings
            bsz = frames_fea.size()[0]
            expand_text_fea = sequence_output.repeat(1, bsz, 1).view(bsz*bsz, 1,768).expand(-1,seq_length,-1)
            expand_frames_fea = frames_fea.repeat(bsz, 1, 1)
            video_mask = video_mask.repeat(bsz,1)                        

            expand_frames_fea = expand_frames_fea.permute(1,0,2)
            expand_text_fea = expand_text_fea.permute(1,0,2)

            visual_output = self.transformerVision(expand_text_fea, expand_frames_fea,expand_frames_fea,video_mask.bool()) 

            visual_output = visual_output.permute(1, 0, 2)                

        elif sim_header == 'videoTransVision':
            seq_length = frames_fea.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=frames_fea.device)
            position_ids = position_ids.unsqueeze(0).expand(frames_fea.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            frames_fea = frames_fea + frame_position_embeddings

            bsz = frames_fea.size()[0]
            expand_text_fea = text_fea.unsqueeze(1).repeat(1, bsz, 1, 1).view(-1, text_fea.size()[1], text_fea.size()[2])                                  
            expand_frames_fea = frames_fea.repeat(bsz, 1, 1)
            video_mask = video_mask.repeat(bsz,1)                        
            expand_att_mask = attention_mask.unsqueeze(1).repeat(1, bsz, 1).view(-1, attention_mask.size()[1])               

            expand_frames_fea = expand_frames_fea.permute(1,0,2)
            expand_text_fea = expand_text_fea.permute(1,0,2)

            visual_output = self.transformerVision(expand_frames_fea, expand_text_fea, expand_text_fea, expand_att_mask.bool())

            visual_output = visual_output + expand_frames_fea
            visual_output = visual_output.permute(1, 0, 2)                

        elif sim_header == 'weightFrame':

            bsz = frames_fea.size()[0]

            expand_frames_fea = frames_fea.repeat(bsz, 1, 1)                                                 

            video_mask = video_mask.repeat(bsz,1)

            expand_sequence_output = sequence_output.expand(bsz, bsz, 768).contiguous().view(-1, 768)                                 

            prob1 = torch.bmm(expand_sequence_output.unsqueeze(1), expand_frames_fea.permute(0, 2, 1)).squeeze(1)                                              
            prob2 = torch.softmax(prob1,dim=-1)
            prob = prob2.unsqueeze(2).expand(-1,-1,frames_fea.size()[-1])                        
            visual_output = expand_frames_fea * prob                             

        elif sim_header == 'weightPrior':
            weight = torch.softmax(self.video_weight_fc(frames_fea).squeeze(2),dim=-1)          
            visual_output = torch.bmm(weight.unsqueeze(1), frames_fea)           

        else:
            raise NotImplementedError()

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        if sim_header=='textTransVision':
            visual_output = self._mean_pooling_for_similarity_visual(visual_output, expand_att_mask)
        elif sim_header=='xpool' or sim_header=='xseq':
            pass
        else:
            visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()

        if sim_header in ['videoTransVision','weightFrame','textTransVision','clsTransVision','xpool','xseq']:
            visual_output = visual_output.view(bsz,bsz,-1)

            retrieve_logits = torch.bmm(sequence_output.unsqueeze(1), visual_output.transpose(1, 2)).squeeze(1)

            retrieve_logits = logit_scale * retrieve_logits
        elif sim_header =='biaffine':
            intermediate = torch.matmul(sequence_output,self.U1)
            intermediate = torch.matmul(intermediate,visual_output.T)
            retrieve_logits = torch.matmul(sequence_output,self.U2) +intermediate
            retrieve_logits = logit_scale * retrieve_logits

        else:
            retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
            if sim_header == 'weightPrior':
                video_self_logits = logit_scale * torch.matmul(visual_output, visual_output.t())
                return retrieve_logits, video_self_logits

        return retrieve_logits,None

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text                                         
        split_size = [step_size] * (b_text // step_size)       
        release_size = b_text - sum(split_size)     
        if release_size > 0:
            split_size += [release_size]

        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)
            cross_output, pooled_output, concat_mask =\
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)
            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output,attention_mask,visual_output, video_mask,video_fea=None,text_fea=None, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()

        if self.interaction == 'wti' or self.interaction == 'ti':
            if self.cdcr == 0:
                retrieve_logits, _, _ = self.wti_interaction(sequence_output, visual_output, attention_mask, video_mask)
                return retrieve_logits, contrastive_direction            
            else:
                retrieve_logits, _, cdcr_loss = self.wti_interaction(sequence_output, visual_output, attention_mask, video_mask)
                return retrieve_logits, contrastive_direction, cdcr_loss

        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf", "videoTransVision","weightFrame","weightPrior","biaffine","textTransVision","clsTransVision",'xpool','xseq']
            retrieve_logits, video_self_logits = self._loose_similarity(sequence_output, attention_mask, visual_output,video_mask, video_fea,text_fea,sim_header=self.sim_header)
        else:
            video_self_logits = None
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(text_fea, video_fea, attention_mask, video_mask)
        return retrieve_logits, contrastive_direction,video_self_logits           

    def wti_interaction(self, text_feat, video_feat, text_mask, video_mask):
        text_feat, video_feat = text_feat.contiguous(), video_feat.contiguous()
        if self.training and torch.cuda.is_available():                    
            text_feat = allgather(text_feat, self.task_config)
            video_feat = allgather(video_feat, self.task_config)
            text_mask = allgather(text_mask, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            torch.distributed.barrier()              
        if self.interaction == 'wti':
            text_weight = self.text_weight_fc(text_feat).squeeze(2)                          
            text_weight.masked_fill_(torch.tensor((1 - text_mask), dtype=torch.bool), float("-inf"))
            text_weight = torch.softmax(text_weight, dim=-1)           

            video_weight = self.video_weight_fc(video_feat).squeeze(2)                         
            video_weight.masked_fill_(torch.tensor((1 - video_mask), dtype=torch.bool), float("-inf"))
            video_weight = torch.softmax(video_weight, dim=-1)           

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
        text_sum = text_mask.sum(-1)
        video_sum = video_mask.sum(-1)

        if self.interaction == 'ti':                          
            t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)               
            v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)               
            t2v_logits = torch.sum(t2v_logits, dim=2) / (text_sum.unsqueeze(1))
            v2t_logits = torch.sum(v2t_logits, dim=2) / (video_sum.unsqueeze(0))
            retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        elif self.interaction == 'wti':                                   
            t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)               
            t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

            v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)               
            v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
            retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        if self.training:
            logit_scale = self.clip.logit_scale.exp()
            retrieve_logits = logit_scale * retrieve_logits

            if self.cdcr == 1:

                _text_feat = text_feat[torch.arange(text_feat.shape[0]),
                                 torch.randint_like(text_sum, 0, 10000) % text_sum, :]
                _video_feat = video_feat[torch.arange(video_feat.shape[0]),
                               torch.randint_like(video_sum, 0, 10000) % video_sum, :]
                z_a_norm = (_text_feat - _text_feat.mean(0)) / _text_feat.std(0)           
                z_b_norm = (_video_feat - _video_feat.mean(0)) / _video_feat.std(0)           

                B, D = z_a_norm.shape
                c = torch.einsum('ac,ad->cd', z_a_norm, z_b_norm) / B       

                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].pow_(2).sum()
                cdcr_loss = (on_diag * self.cdcr_alpha1 + off_diag * self.cdcr_alpha2)
                return retrieve_logits, retrieve_logits.T, cdcr_loss
            elif self.cdcr == 2:

                max_idx1 = max_idx1[torch.arange(max_idx1.shape[0]), torch.arange(max_idx1.shape[1])]
                max_idx2 = max_idx2[torch.arange(max_idx2.shape[0]), torch.arange(max_idx2.shape[1])]

                max_t_feat = text_feat[torch.arange(max_idx2.shape[0]).repeat_interleave(max_idx2.shape[1]),
                                       max_idx2.flatten()]
                max_v_feat = video_feat[torch.arange(max_idx1.shape[0]).repeat_interleave(max_idx1.shape[1]),
                                        max_idx1.flatten()]

                t_feat = text_feat.reshape(-1, text_feat.shape[-1])
                t_mask = text_mask.flatten().type(torch.bool)
                v_feat = video_feat.reshape(-1, text_feat.shape[-1])
                v_mask = video_mask.flatten().type(torch.bool)
                t_feat = t_feat[t_mask]
                v_feat = v_feat[v_mask]
                max_t_feat = max_t_feat[v_mask]
                max_v_feat = max_v_feat[t_mask]

                z_a_norm = (t_feat - t_feat.mean(0)) / t_feat.std(0)             
                z_b_norm = (max_v_feat - max_v_feat.mean(0)) / max_v_feat.std(0)             

                x_a_norm = (v_feat - v_feat.mean(0)) / v_feat.std(0)             
                x_b_norm = (max_t_feat - max_t_feat.mean(0)) / max_t_feat.std(0)             

                N, D = z_a_norm.shape
                c1 = torch.einsum('ac,ad->cd', z_a_norm, z_b_norm) / N       
                N, D = x_a_norm.shape
                c2 = torch.einsum('ac,ad->cd', x_a_norm, x_b_norm) / N       
                c = (c1 + c2) / 2.0

                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].pow_(2).sum()
                cdcr_loss = (on_diag * self.cdcr_alpha1 + off_diag * self.cdcr_alpha2)
                return retrieve_logits, retrieve_logits.T, cdcr_loss
            elif self.cdcr == 3:

                max_idx1 = max_idx1[torch.arange(max_idx1.shape[0]), torch.arange(max_idx1.shape[1])]
                max_idx2 = max_idx2[torch.arange(max_idx2.shape[0]), torch.arange(max_idx2.shape[1])]

                max_t_feat = text_feat[torch.arange(max_idx2.shape[0]).repeat_interleave(max_idx2.shape[1]),
                                       max_idx2.flatten()].squeeze(1)
                max_v_feat = video_feat[torch.arange(max_idx1.shape[0]).repeat_interleave(max_idx1.shape[1]),
                                        max_idx1.flatten()].squeeze(1)

                t_feat = text_feat.reshape(-1, text_feat.shape[-1])
                t_mask = text_mask.flatten().type(torch.bool)
                v_feat = video_feat.reshape(-1, video_feat.shape[-1])
                v_mask = video_mask.flatten().type(torch.bool)
                t_feat = t_feat[t_mask]
                v_feat = v_feat[v_mask]
                max_t_feat = max_t_feat[v_mask]
                max_v_feat = max_v_feat[t_mask]
                text_weight = text_weight.flatten()[t_mask]
                video_weight = video_weight.flatten()[v_mask]

                z_a_norm = (t_feat - t_feat.mean(0)) / t_feat.std(0)             
                z_b_norm = (max_v_feat - max_v_feat.mean(0)) / max_v_feat.std(0)             

                x_a_norm = (v_feat - v_feat.mean(0)) / v_feat.std(0)             
                x_b_norm = (max_t_feat - max_t_feat.mean(0)) / max_t_feat.std(0)             

                N, D = z_a_norm.shape
                B = text_feat.shape[0]
                c1 = torch.einsum("acd,a->cd", torch.einsum('ac,ad->acd', z_a_norm, z_b_norm),
                                  text_weight) / B       
                c2 = torch.einsum("acd,a->cd", torch.einsum('ac,ad->acd', x_a_norm, x_b_norm),
                                  video_weight) / B       
                c = (c1 + c2) / 2.0

                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].pow_(2).sum()
                cdcr_loss = (on_diag * self.cdcr_alpha1 + off_diag * self.cdcr_alpha2)
                return retrieve_logits, retrieve_logits.T, cdcr_loss
            else:
                return retrieve_logits, retrieve_logits.T, 0.0
        else:
            return retrieve_logits, retrieve_logits.T, 0.0
