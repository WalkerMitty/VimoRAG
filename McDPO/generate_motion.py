import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

# import lightning as L
import torch
import numpy as np
import json
import models.vqvae as vqvae

from options import option
import imageio
from utils.evaluate_visual import plot
from visualization.render import render
warnings.filterwarnings('ignore')

args = option.get_args_parser()


def main(
    quantize: Optional[str] = None,
    dtype: str = "float32",
    max_new_tokens: int = 200,
    top_k: int = 200,
    temperature: float = 0.1,
    accelerator: str = "auto",
) -> None:


    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)
    print ('loading checkpoint from {}'.format(args.vqvae_path))
    ckpt = torch.load(args.vqvae_path, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.cuda()

    with open(args.generated_file,"r") as f:
        data = json.load(f)
    for key,value in data.items():
        output = value    
    tokens = torch.tensor([int(token) for token in output.split(',')]).cuda()
    generated_pose, img = plot(tokens, net, args.dataname)
    os.makedirs(args.out_dir, exist_ok=True)
    # np.save(os.path.join(args.out_dir, 'demo.npy'), generated_pose)
    # print(generated_pose.shape)
    # exit()
    imageio.mimsave(os.path.join(args.out_dir, 'demo.gif'), np.array(img), fps=20)
    # imageio.mimsave(os.path.join(args.out_dir, 'demo.mp4'), np.array(img), fps=20, codec='libx264', quality=10)
    # exit()
    if args.render:
        print("Rendering...")
        render(generated_pose, 'demo', outdir=args.out_dir)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
