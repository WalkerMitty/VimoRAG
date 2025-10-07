from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn
import torch.nn.functional as F


from modules.until_module import PreTrainedModel, AllGather, CrossEn, LossWeight
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip
from modules.action_clip import build_action_model,build_verb_model
from modules.module_clip import CLIP, convert_weights
from modules import clip_evl


logger = logging.getLogger(__name__)
allgather = AllGather.apply
# from einops import rearrange



class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None
        self.temp_count = 0

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
        # import ipdb;ipdb.set_trace()
        ############# add ################
        clip_state_dict = clip_state_dict['state_dict'] if 'state_dict' in clip_state_dict else clip_state_dict
        for key, val in clip_state_dict.items():
            if key not in state_dict:
                state_dict[key] = val.clone()
            new_key = key.replace('clip.', '')
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()
        ############## add ####################
        
        
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        # import ipdb;ipdb.set_trace()
        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE] 
        if task_config.train_tower not in ['action','event'] and model.linear_patch == "3d":
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
        sim_matrix = sim_matrix * F.softmax(sim_matrix/temp, dim=0)*len(sim_matrix) #With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss
    
class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        ### comment this for now
        # assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        # self._stage_one = True
        # self._stage_two = False

        # show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        # self.loose_type = True
        # if self._stage_one and check_attr('loose_type', self.task_config):
        #     self.loose_type = True
        #     show_log(task_config, "Test retrieval by loose type.")
        embed_dim = clip_state_dict["text_projection"].shape[1] #768
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        if task_config.train_tower=='object':
            # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
            ##############add###################
            if 'clip.visual.proj' in clip_state_dict:
                new_dict = {}
                for k, v in clip_state_dict.items():
                    new_k = k.replace('clip.', '')
                    new_dict[new_k] = v.clone()
            
                clip_state_dict = new_dict
            ##############add###################
            
            vit = "visual.proj" in clip_state_dict
            assert vit
            if vit:
                vision_width = clip_state_dict["visual.conv1.weight"].shape[0] #1024
                vision_layers = len(
                    [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
                vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1] #14
                grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5) #16
                image_resolution = vision_patch_size * grid_size #224
            else:
                counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                                [1, 2, 3, 4]]
                vision_layers = tuple(counts)
                vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
                output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
                vision_patch_size = None
                assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
                image_resolution = output_width * 32


            context_length = clip_state_dict["positional_embedding"].shape[0] #77
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0] #49408

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

            # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
            cut_top_layer = 0
            show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
            
            #### paramters for DRL, we do not use them for now ####
            self.interaction = task_config.interaction if hasattr(task_config, 'interaction') else 'no'
            self.wti_arch = task_config.wti_arch if hasattr(task_config, 'wti_arch') else 0
            self.mlp_layer = task_config.mlp_layer if hasattr(task_config, 'mlp_layer') else 0
            self.cdcr = task_config.cdcr if hasattr(task_config, 'cdcr') else 0
            if hasattr(task_config, "clip_evl") and task_config.clip_evl == True: 
                # import ipdb;ipdb.set_trace()
                self.clip, _ = clip_evl.load(task_config.pretrained_path, t_size=task_config.max_frames, mergeclip=task_config.mergeclip, mergeweight=task_config.mergeweight, clip_state_dict=clip_state_dict)
                # import ipdb;ipdb.set_trace()
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
            cross_config.max_position_embeddings = context_length
        elif task_config.train_tower=='action':
            self.verb_model = build_verb_model(task_config.pretrained_path,task_config.verb_model)
            self.action_model = build_action_model(task_config.action_model)
            self.frame_position_embeddings = nn.Embedding(task_config.max_frames, embed_dim)
            # import pdb;pdb.set_trace()
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )

        elif task_config.train_tower == 'event':
            # self.event_extractor = build_verb_model(task_config.pretrained_path,task_config.verb_model)
            #v1
            self.event_extractor = nn.Linear(768,2)

        else:
            raise ValueError()


        self.loss_fct = CrossEn()
        
        # self.loss_dsl = dual_softmax_loss()
    def get_weight(self,input_ids, sequence_output,sequence_output_verb):
        self.temp_count +=1
        if self.training:
            # visual_output = allgather(visual_output, self.task_config)
            input_ids = allgather(input_ids,self.task_config)

            torch.distributed.barrier()
        bs_pair = input_ids.size(0)
        bs_pair_second = sequence_output.size(0)
        tau_hidden = self.event_extractor(input_ids).float() #[bsz, 768]
        tau_hidden = tau_hidden / tau_hidden.norm(dim=-1, keepdim=True)
        p_hidden = sequence_output_verb.squeeze(1) #[bsz,768]
        a_hidden = sequence_output.squeeze(1) #[bsz,768]
        # import ipdb;ipdb.set_trace()
        p_sim = torch.matmul(tau_hidden,p_hidden.T).view(1,-1) #[1,bsz*bsz]
        a_sim = torch.matmul(tau_hidden,a_hidden.T).view(1,-1) #[1,bsz*bsz]

        score = F.softmax(torch.cat((p_sim,a_sim),dim=0),dim=0)
        if self.temp_count %10==0:
            print('verb weight:  ',score[0].mean().item())
        p_score = score[0].view(bs_pair,bs_pair_second)
        a_score = score[1].view(bs_pair,bs_pair_second)
        # import ipdb;ipdb.set_trace()
        return p_score,a_score
    def get_weight_action(self,input_ids,sequence_output_verb):
        self.temp_count +=1
        if self.training:
            input_ids = allgather(input_ids,self.task_config)
            torch.distributed.barrier()
        bs_pair = input_ids.size(0)
        a_hidden = sequence_output_verb.squeeze(1) #[bsz,768]
        bs_v = a_hidden.size(0)
        score = F.softmax(self.event_extractor(a_hidden),dim=1) #[bsz_v,2]


        p_score = score[:,0].expand(bs_pair,bs_v)
        a_score = score[:,1].expand(bs_pair,bs_v)
        # import ipdb;ipdb.set_trace()
        return p_score,a_score
        
    def forward(self, input_ids,attention_mask, video, video_mask,key_points=None,init_kp=None,key_mask=None,motion_mask=None,objectmodel=None,actionmodel=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        if key_points is not None:
            key_points = key_points.squeeze(1) #[bsz,16,37]. or [bsz,16,17,3]
            if self.task_config.action_model=='wham':
                init_kp = init_kp.squeeze(1)
                key_mask = key_mask.squeeze(1)
        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts
        
        if self.task_config.train_tower == 'action':
            sequence_output_verb,motion_output = self.get_verb_action_output(input_ids,attention_mask, key_points,init_kp,key_mask,shaped=True)
        elif self.task_config.train_tower == 'object':
            
            sequence_output, visual_output = self.get_sequence_visual_output(input_ids,video, video_mask, shaped=True, video_frame=video_frame)
        else:
            sequence_output_verb,motion_output = actionmodel.get_verb_action_output(input_ids,attention_mask, key_points,init_kp,key_mask,shaped=True)
            sequence_output, visual_output = objectmodel.get_sequence_visual_output(input_ids,video, video_mask, shaped=True, video_frame=video_frame)
        if self.training:
            loss = 0.
            # loss_action = 0.
            if self.task_config.train_tower == 'action':
                sim_matrix = self.get_similarity_logits(sequence_output_verb, motion_output,motion_mask,shaped=True,action=True)
            elif self.task_config.train_tower == 'object':
                sim_matrix = self.get_similarity_logits(sequence_output,visual_output,video_mask,shaped=True, action=False)
            else:
                
                sim_matrix_action,new_motion_output = actionmodel.get_similarity_logits(sequence_output_verb, motion_output,motion_mask,shaped=True,action=True)
                sim_matrix_object,new_visual_output = objectmodel.get_similarity_logits(sequence_output,visual_output,video_mask,shaped=True,action=False)
                # import ipdb;ipdb.set_trace()
                # weight_action,weight_object = self.get_weight(input_ids, new_visual_output,new_motion_output)
                weight_action,weight_object = self.get_weight_action(input_ids, new_motion_output)
                sim_matrix = sim_matrix_action * weight_action + sim_matrix_object * weight_object
            # import ipdb;ipdb.set_trace()
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            loss += sim_loss


            return loss
        else:
            return None

    def get_sequence_output(self, input_ids,shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])

        bs_pair = input_ids.size(0)

        sequence_hidden = self.clip.encode_text(input_ids).float()
            #>>>>
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        # return sequence_hidden
        return sequence_hidden

    def get_verb_output(self, input_ids,attention_mask,shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            # token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])

        bs_pair = input_ids.size(0)
        # import ipdb;ipdb.set_trace()
        sequence_hidden = self.verb_model(input_ids,attention_mask).float()

        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
        #video: [bsz*12, 3, 224,224], video_mask [bsz,12]
        bs_pair = video_mask.size(0)
        if self.clip_evl:
            if len(video.size()) == 4:
                # [b, t, c, h, w]
                video = video.view(bs_pair, -1, video.size(-3), video.size(-2), video.size(-1))
                video = video.permute(0, 2, 1, 3, 4).contiguous()
            # [N, 1, d], [L, N, T, d].   video:[32,3, 12, 224,224]
            visual_output,_ = self.clip.encode_video(video, return_all_feats=True)
            # visual_output = visual_output.float(). evl_output: [bsz, 768].  visual_output: [257, bsz, 12, 1024]
            # visual_hidden = evl_output.float()
            #our add <<.. the below line
            visual_hidden = visual_output.float()
            # video_fea = video_fea.float()

        else:
            visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1)) 
        #bsz, 1, 768
        # return visual_hidden,visual_output
        return visual_hidden


    def get_action_output(self, key_points,init_kp,key_mask):
        # import ipdb;ipdb.set_trace()
        # key_points [bsz,16,37] init_Kp [bsz,1,88] key_mask [bsz,16,17]
        if self.task_config.action_model=='wham':
            motion_output = self.action_model(key_points,init_kp,key_mask)
            motion_output = motion_output.float()
        elif self.task_config.action_model=='motionbert':
            motion_output = self.action_model(key_points) #[bsz,16,17,768]
        else:
            raise ValueError()
        return motion_output

    def get_sequence_visual_output(self, input_ids, video, video_mask, shaped=False, video_frame=-1):
        # import ipdb;ipdb.set_trace()
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            # token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            # attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
        sequence_output = self.get_sequence_output(input_ids, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)     

        return sequence_output, visual_output
    def get_verb_action_output(self,input_ids, attention_mask,key_points,init_kp,key_mask,shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1,attention_mask.shape[-1])
            key_points = key_points.squeeze(1)
            if self.task_config.action_model=='wham':
                init_kp = init_kp.squeeze(1)
                key_mask = key_mask.squeeze(1)

        sequence_output = self.get_verb_output(input_ids,attention_mask,shaped=True)
        visual_output = self.get_action_output(key_points,init_kp,key_mask)
        return sequence_output,visual_output        



    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1) #bsz,12,1
        visual_output = visual_output * video_mask_un #[bsz, 12, 768]
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)  #bsz, 1
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out
    def _loose_similarity_action(self,sequence_output,visual_output,video_mask):

                # Sequential type: Transformer Encoder
        # import ipdb;ipdb.set_trace()

        visual_output_original = visual_output.clone() #[bsz,16,768] video mask: [bsz,1,16]
        video_mask = video_mask.squeeze(1) #[bsz,16] 

        seq_length = visual_output.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1) #[bsz,16,hidden]
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        visual_output = visual_output + frame_position_embeddings

        extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
        # import ipdb;ipdb.set_trace()
        visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
        visual_output = self.transformerClip(visual_output, extended_video_mask)
        visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
        visual_output = visual_output + visual_output_original

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        #[bsz, 768]

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        # print('size2',sequence_output.size()). [128, 768]
        # exit()

        retrieve_logits = 100.0 * torch.matmul(sequence_output, visual_output.t())

        return retrieve_logits,visual_output

    def _loose_similarity(self, sequence_output,visual_output,video_mask):
 
        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        #[bsz, 768]

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        # print('size2',sequence_output.size()). [128, 768]
        # exit()
        logit_scale = self.clip.logit_scale.exp()

        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        
        return retrieve_logits,visual_output


    def get_similarity_logits(self, sequence_output,visual_output, video_mask, shaped=False,action=False):
        if shaped is False:
            # attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if action:
            retrieve_logits,new_visual_embed = self._loose_similarity_action(sequence_output, visual_output,video_mask)
        else:
            retrieve_logits,new_visual_embed = self._loose_similarity(sequence_output,visual_output,video_mask)
        if self.task_config.train_tower=='event':
            return retrieve_logits,new_visual_embed
        return retrieve_logits