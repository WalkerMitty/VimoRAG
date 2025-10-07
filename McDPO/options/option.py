import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    # parser.add_argument('--pretrained_llama', type=str, default="13B")
    # parser.add_argument('--vqvae_pth', type=str, default='../resources/pretrained_vqvae/t2m.pth', help='path to the pretrained vqvae pth')

    ## vqvae
    parser.add_argument("--code_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb_code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq_act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    # parser.add_argument('--seed', default=123, type=int, help='seed for initializing vqvae training.')
    parser.add_argument('--window_size', type=int, default=64, help='training motion length')
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## visualization
    parser.add_argument("--render", action='store_true', help='render smpl')

    ## evaluate_final.py
    parser.add_argument("--model_path", default='',type=str)
    parser.add_argument("--video_path", default='dataset/HumanML3D_videos/videos',type=str)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--vqvae_path",default='../resources/pretrained_vqvae/t2m.pth',type=str)
    parser.add_argument("--start",default=-1,type=int,help='if start==-1, single gpu only,  start id of eval batch')
    parser.add_argument("--end",default=-1,type=int,help='only works when start!=-1, -1 means the last sample')
    parser.add_argument('--out_dir', type=str, default='eval_output', help='output directory, if start!=-1, then some intermediate results will be saved here, the prefix is ``$start_id_``')
    parser.add_argument('--seed', type=int, default=2024, help='multi gpu mode, the seed must be consistent!')
    parser.add_argument("--retrieval_result", default=None,type=str)
    parser.add_argument("--text_only", action='store_true')
    parser.add_argument("--demo_inference", action='store_true')

    ## evalute_for_generated_results.py
    parser.add_argument("--generated_file", default=None,type=str)
    parser.add_argument("--motion_encoder", action='store_true')
    parser.add_argument("--lora", action='store_true')
    parser.add_argument('--llm_seed', type=int, default=2024, help='only control the llm generation')
    parser.add_argument("--split", type=str, default='val', choices = ['train','val','test'])
    parser.add_argument("--fid_weight", default=0.5, type=float)
    parser.add_argument("--match_weight", default=0.5, type=float)
    parser.add_argument("--dpo_selection", action='store_true')
    parser.add_argument("--sft_file", default=None,type=str)
    parser.add_argument("--dpo_file", default=None,type=str)
    
    # merge lora
    parser.add_argument("--merge_lora", action='store_true')
    parser.add_argument("--model_base", default='',type=str)
    return parser.parse_args()
