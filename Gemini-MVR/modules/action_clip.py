import torch
from torch import nn
from modules.clip_evl.model import Transformer,LayerNorm
import numpy as np
from transformers import BertModel
from .DSTformer import DSTformer 
from functools import partial

# action_model_type = 'motionbert' #or wham

class VerbModel(nn.Module):
    def __init__(self, embed_dim: int,context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,):
        super(VerbModel, self).__init__()
        self.context_length = context_length
        # import pdb;pdb.set_trace()
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            use_checkpoint=False,
            checkpoint_num=[24, 100],
        )
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    @property
    def dtype(self):
        return self.token_embedding.weight.dtype
    def forward(self, text, masked_indices=None, return_all_feats=False):
        # import ipdb;ipdb.set_trace()
        # assert (text.max(dim=-1)[0] + 1 == self.token_embedding.num_embeddings).all(), \
        #     "The last token of each sentence should be eot_token, check the input"

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # x[torch.arange(x.shape[0]), text.argmax(dim=-1)] += self.eot_token_embedding

        # if masked_indices is not None:
        #     x[masked_indices] = self.text_mask_embedding

        x = x + self.positional_embedding.type(self.dtype)
        #[bsz, 77, 768]
        x = x.permute(1, 0, 2)  # NLD -> LND.
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        feats = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        if self.text_projection is not None:
            feats = feats @ self.text_projection

                #<<< our add:
            x = x @ self.text_projection
                #>>

        if return_all_feats:
            return feats, x

        return feats

class VerbModelBert(nn.Module):
    def __init__(self,path):
        super(VerbModelBert, self).__init__()
        self.model = BertModel.from_pretrained(path)
    def forward(self,input_ids,attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        cls_hidden_states = last_hidden_states[:, 0, :]
        return cls_hidden_states

def build_verb_model(pretrained_path,verb_model):
    '''
    '''
    if verb_model=='bert':
        verbmodel = VerbModelBert('llms/bert-base-uncased')
    elif verb_model == 'internvideo':
        init_state_dict = torch.load(pretrained_path, map_location='cpu')['state_dict']
        state_dict = {}
        for k, v in init_state_dict.items():
            k = k.replace('clip.','')
            state_dict[k] = v
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        verbmodel = VerbModel(embed_dim,context_length,vocab_size,transformer_width,transformer_heads,transformer_layers)
        # import ipdb;ipdb.set_trace()
        msg = verbmodel.load_state_dict(state_dict, strict=False)
        print(msg)
    else:
        raise ValueError()
    return verbmodel


class ActionModelBERT(nn.Module):
    def __init__(self,encoder_path):
        super(ActionModelBERT,self).__init__()
        
        self.encoder = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, 
                                   depth=5, num_heads=8, mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=243, num_joints=17)
        checkpoint = torch.load(encoder_path)
        new_check= {}
        for key,value in checkpoint['model_pos'].items():
            new_check[key.replace('module.','')] = value
        # import ipdb;ipdb.set_trace()
        del new_check['head.bias']
        del new_check['head.weight'] #we do not use the head layer
        self.encoder.load_state_dict(new_check, strict=True)
        self.projector = nn.Linear(17*512,768)
    def forward(self,input):
        bsz = input.size()[0]
        n_frames = input.size()[1]
        # import ipdb;ipdb.set_trace()
        output = self.encoder(input,return_rep=True).view(bsz,n_frames,-1)
        return self.projector(output)




class ActionModel(nn.Module):
    def __init__(self,encoder_path):
        super(ActionModel,self).__init__()
        self.encoder = PretrainMotionEncoder(encoder_path)
        self.projector = nn.Linear(563,768)
    def forward(self,x,init_kp,key_mask):
        return self.projector(self.encoder(x,init_kp,key_mask))

def build_action_model(action_model):
    if action_model=='wham':
        model = ActionModel('WHAM/checkpoints/wham_vit_bedlam_w_3dpw.pth.tar')
    else:
        model = ActionModelBERT('../resources/MotionBERT/checkpoint/pretrain/latest_epoch.bin')
    return model.eval()

class PretrainMotionEncoder(nn.Module):
    def __init__(self,model_path):
        super(PretrainMotionEncoder, self).__init__()
        self.motion_encoder = self.build_motion_encoder(model_path)
    def forward(self,x, init,mask):
        _,output = self.motion_encoder(x, init,mask)
        return output

    def build_motion_encoder(self,model_path):
        n_joints = 17
        # in_dim = n_joints * 2 + 3
        model = MotionEncoder(in_dim=37, 
                            d_embed=512,
                            pose_dr=0.15,
                            rnn_type='LSTM',
                            n_layers=3,
                            n_joints=n_joints)
        # model.load(model_path)
        checkpoint = torch.load(model_path)
        # import ipdb;ipdb.set_trace()
        # ignore_keys = ['smpl.body_pose', 'smpl.betas', 'smpl.global_orient', 'smpl.J_regressor_extra', 'smpl.J_regressor_eval']
        # model_state_dict = {k: v for k, v in checkpoint['model'].items() if k not in ignore_keys}
        model_state_dict = {k.replace('motion_encoder.',''): v for k, v in checkpoint['model'].items() if 'motion_encoder' in k}
        model.load_state_dict(model_state_dict, strict=True)
        # import ipdb;ipdb.set_trace()
        return model
    def freeze(self,):
        for p in self.motion_encoder.parameters():
            p.requires_grad = False
        

class Regressor(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dims, init_dim, layer='LSTM', n_layers=2, n_iters=1):
        super().__init__()
        self.n_outs = len(out_dims)

        self.rnn = getattr(nn, layer.upper())(
            in_dim + init_dim, hid_dim, n_layers, 
            bidirectional=False, batch_first=True, dropout=0.3)

        for i, out_dim in enumerate(out_dims):
            setattr(self, 'declayer%d'%i, nn.Linear(hid_dim, out_dim))
            nn.init.xavier_uniform_(getattr(self, 'declayer%d'%i).weight, gain=0.01)

    def forward(self, x, inits, h0):
        xc = torch.cat([x, *inits], dim=-1)
        xc, h0 = self.rnn(xc, h0)

        preds = []
        for j in range(self.n_outs):
            out = getattr(self, 'declayer%d'%j)(xc)
            preds.append(out)

        return preds, xc, h0
    
    
class NeuralInitialization(nn.Module):
    def __init__(self, in_dim, hid_dim, layer, n_layers):
        super().__init__()

        out_dim = hid_dim
        self.n_layers = n_layers
        self.num_inits = int(layer.upper() == 'LSTM') + 1
        out_dim *= self.num_inits * n_layers

        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim * self.n_layers)
        self.linear3 = nn.Linear(hid_dim * self.n_layers, out_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        b = x.shape[0]

        out = self.linear3(self.relu2(self.linear2(self.relu1(self.linear1(x)))))
        out = out.view(b, self.num_inits, self.n_layers, -1).permute(1, 2, 0, 3).contiguous()

        if self.num_inits == 2:
            return tuple([_ for _ in out])
        return out[0]


class MotionEncoder(nn.Module):
    def __init__(self, 
                 in_dim, 
                 d_embed,
                 pose_dr,
                 rnn_type,
                 n_layers,
                 n_joints):
        super().__init__()
        
        self.n_joints = n_joints
        
        self.embed_layer = nn.Linear(in_dim, d_embed)
        self.pos_drop = nn.Dropout(pose_dr)
        
        # Keypoints initializer
        self.neural_init = NeuralInitialization(88, d_embed, rnn_type, n_layers)
        # 3d keypoints regressor
        self.regressor = Regressor(
            d_embed, d_embed, [n_joints * 3], n_joints * 3, rnn_type, n_layers)
    def preprocess(self,x,mask):
        self.b, self.f = x.shape[:2]
        self_mask_embedding = nn.Parameter(torch.zeros(1, 1, self.n_joints, 2)).to(x.device).to(x.dtype)
        # Treat masked keypoints
        mask_embedding = mask.unsqueeze(-1) * self_mask_embedding
        _mask = mask.unsqueeze(-1).repeat(1, 1, 1, 2).reshape(self.b, self.f, -1)
        _mask = torch.cat((_mask, torch.zeros_like(_mask[..., :3])), dim=-1)
        _mask_embedding = mask_embedding.reshape(self.b, self.f, -1)
        _mask_embedding = torch.cat((_mask_embedding, torch.zeros_like(_mask_embedding[..., :3])), dim=-1)
        x[_mask] = 0.0
        x = x + _mask_embedding
        return x
    def forward(self, x, init,mask):
        
        self.b, self.f = x.shape[:2]
        x = self.preprocess(x,mask)
        # import ipdb;ipdb.set_trace()
        x = self.embed_layer(x.reshape(self.b, self.f, -1))
        x = self.pos_drop(x)
        
        h0 = self.neural_init(init)
        pred_list = [init[..., :self.n_joints * 3]]
        motion_context_list = []
        
        for i in range(self.f):
            (pred_kp3d, ), motion_context, h0 = self.regressor(x[:, [i]], pred_list[-1:], h0)
            motion_context_list.append(motion_context)
            pred_list.append(pred_kp3d)
            
        pred_kp3d = torch.cat(pred_list[1:], dim=1).view(self.b, self.f, -1, 3)
        motion_context = torch.cat(motion_context_list, dim=1)
        
        # Merge 3D keypoints with motion context
        motion_context = torch.cat((motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
        return pred_kp3d, motion_context
