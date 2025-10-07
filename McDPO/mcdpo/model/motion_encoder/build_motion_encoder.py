import torch
import numpy as np
from torch import nn
from functools import partial
from .DSTformer import DSTformer
def build_motion_encoder(model_path):
    model = PretrainMotionEncoder(model_path)
    model = model.to(torch.device('cuda'))
    model_without_ddp = model
    model_without_ddp = model_without_ddp.to(torch.float32)

    model_without_ddp.eval()
    return (model_without_ddp)
class PretrainMotionEncoder(nn.Module):
    def __init__(self,model_path):
        super(PretrainMotionEncoder, self).__init__()
        self.motion_encoder = self.build_motion_encoder(model_path)
        # import ipdb;ipdb.set_trace()
        self.freeze()
    def forward(self,x):
        # import ipdb;ipdb.set_trace()
        bsz = x.size()[0]
        n_frames = x.size()[1]
        output = self.motion_encoder(x, return_rep = True).view(bsz,n_frames,-1)
        return output
    def load_model(self,path):
        checkpoint = torch.load(path)
        new_check= {}
        for key,value in checkpoint['model_pos'].items():
            new_check[key.replace('module.','motion_encoder.')] = value
        del new_check['motion_encoder.head.bias']
        del new_check['motion_encoder.head.weight'] #we do not use the head layer
        # import ipdb;ipdb.set_trace()
        self.load_state_dict(new_check, strict=True)

    # def build_motion_encoder(self,model_path):
    #     '''
    #     this is for wham
    #     '''
    #     n_joints = 17
    #     # in_dim = n_joints * 2 + 3
    #     model = MotionEncoder(in_dim=37, 
    #                         d_embed=512,
    #                         pose_dr=0.15,
    #                         rnn_type='LSTM',
    #                         n_layers=3,
    #                         n_joints=n_joints)
    #     checkpoint = torch.load(model_path)
    #     model_state_dict = {k.replace('motion_encoder.',''): v for k, v in checkpoint['model'].items() if 'motion_encoder' in k}
    #     model.load_state_dict(model_state_dict, strict=True)

    #     return model
    def build_motion_encoder(self,model_path):
        '''
        this is for motionbert
        '''
        model = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, 
                                    depth=5, num_heads=8, mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                    maxlen=243, num_joints=17)
        checkpoint = torch.load('../resources/MotionBERT/checkpoint/pretrain/latest_epoch.bin')
        new_check= {}
        for key,value in checkpoint['model_pos'].items():
            new_check[key.replace('module.','')] = value
        del new_check['head.bias']
        del new_check['head.weight'] #we do not use the head layer
        model.load_state_dict(new_check, strict=True)
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
