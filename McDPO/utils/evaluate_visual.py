from .motion_process import recover_from_ric
from visualization.plot_3d_global import plot_3d_motion
import torch
import numpy as np
import os 
@torch.no_grad()
def plot(tokens, net, dataname):
    meta_dir = "./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta" if dataname == 't2m' else "./checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
    mean = torch.FloatTensor(np.load(os.path.join(meta_dir, 'mean.npy'))).cuda()
    std = torch.FloatTensor(np.load(os.path.join(meta_dir, 'std.npy'))).cuda()

    pred_pose = net.forward_decoder(tokens)
    pred_pose_denorm = pred_pose * std + mean
    pred_xyz = recover_from_ric(pred_pose_denorm, joints_num=22 if dataname == 't2m' else 21).detach().cpu().numpy()[0]
    img = plot_3d_motion([pred_xyz, None, None])
    return pred_xyz, img