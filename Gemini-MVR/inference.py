from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT
import torch.distributed as dist
import subprocess
from ipdb import set_trace
import torch.nn as nn
from scipy.special import softmax
from tqdm import tqdm
# torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    # parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    # parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    # parser.add_argument('--train_csv', type=str, default='data/MSR-VTT/anns/MSRVTT_train.9k.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/MSR-VTT/anns/MSRVTT_JSFUSION_test.csv', help='')
    # parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    # parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    # parser.add_argument('--features_path', type=str, default='s3://video_pub/MSR-VTT/videos', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=4, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    # parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    # parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    # parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    # parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    #parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                    help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir", default='exps/clip4clip', type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    # parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    # parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    # parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    # parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    # parser.add_argument('--sim_header', type=str, default="meanP",
    #                     choices=["meanP", "seqLSTM", "seqTransf", "tightTransf","videoTransVision","biaffine","weightFrame"],
    #                     help="choice a similarity header.")

    #### CLIP KC/EVL ######
    parser.add_argument("--zeroshot", action='store_true', help="Choose a CLIP version")
    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument("--clip_evl", action='store_true', help="Choose a CLIP version")
    parser.add_argument("--clip_kc", action='store_true', help="Choose a CLIP version")
    parser.add_argument("--use_dsl", action='store_true', help="Choose a CLIP version")
    parser.add_argument("--clip_kc2", action='store_true', help="This is for ViT-B/16")
    parser.add_argument("--clip_kc4", action='store_true', help="This is for ViT-L/14")
    ## weightPrior

    # parser.add_argument("--mlp_layer", type=int, default=1, choices =[1,2,3,4],help="the number of layers of mlp in weightPrior")
    parser.add_argument("--mlp_lr", type=float, default=1e-4, help="the lr of mlp in weightPrior, the lr of attention module in videoTransVision and seqTransf")
    parser.add_argument("--train_tower", type=str, choices = ['action','object','event'],help="the training stage of the hierarchical model, 'action' means only action model is trained, 'event' means freeze the other models, keeping the event extractor tuning ")
    parser.add_argument("--verb_model", type=str, choices=['bert','internvideo'], help="the type of text encoder in verb model")
    parser.add_argument("--action_model", type=str, choices=['wham','motionbert'], help="the type of action encoder")

    ### DRL ###
    parser.add_argument("--interaction", type=str, default='no', help="Choose a CLIP version")
    parser.add_argument("--wti_arch", type=int, default=0, help="Choose a CLIP version")
    parser.add_argument("--cdcr", type=int, default=0, help="Choose a CLIP version")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Choose a CLIP version")
    parser.add_argument("--mergeclip", type=bool, default=False, help="Choose a CLIP version")
    parser.add_argument("--mergeweight", type=float, default=0.5, help="Choose a CLIP version")
    parser.add_argument("--use_capdecoder", type=bool, default=False, help="Choose a CLIP version")
    
    parser.add_argument("--finetuned_path", type=str, default=None, help="the saved model")
    parser.add_argument("--finetuned_path_object", type=str, default=None, help="pretrained model of object for event model")
    parser.add_argument("--finetuned_path_action", type=str, default=None, help="Choose a CLIP version")

    args = parser.parse_args()

    # if args.sim_header == "tightTransf":
    #     args.loose_type = False

    # Check paramenters
    # if args.gradient_accumulation_steps < 1:
    #     raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
    #         args.gradient_accumulation_steps))
    # if not args.do_train and not args.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        #args.rank = int(os.environ['SLURM_PROCID'])
        #args.gpu = args.rank % torch.cuda.device_count()
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = os.environ['SLURM_NTASKS']
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list)
        )
        master_port = os.environ.get('MASTER_PORT', '29498')
        master_port = "29481"
        os.environ['MASTER_PORT'] = master_port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = int(ntasks)
        args.rank = int(proc_id)
        args.gpu = int(proc_id % num_gpus)
        print(f'SLURM MODE: proc_id: {proc_id}, ntasks: {ntasks}, node_list: {node_list}, num_gpus:{num_gpus}, addr:{addr}, master port:{master_port}' )
        
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        # init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    # print('| distributed init (rank {}): {}'.format(
    #     args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

    
def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # world_size = torch.distributed.get_world_size()
    # torch.cuda.set_device(args.local_rank)
    # args.world_size = world_size
    # rank = torch.distributed.get_rank()
    # args.rank = rank
    init_distributed_mode(args)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):


    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=None, task_config=args)

    if args.finetuned_path is not None:
        finetuned_ckpt = torch.load(args.finetuned_path, map_location='cpu')
        model.load_state_dict(finetuned_ckpt)
    # set_trace()

    model.cuda()

    return model

def init_model_for_three(args, device, n_gpu, local_rank):


    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    args.train_tower = 'object'
    object_model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=None, task_config=args)

    if args.finetuned_path_object is not None:
        finetuned_ckpt = torch.load(args.finetuned_path_object, map_location='cpu')
        object_model.load_state_dict(finetuned_ckpt)

    object_model.cuda()
    args.train_tower = 'action'
    action_model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=None, task_config=args)

    if args.finetuned_path_action is not None:
        finetuned_ckpt = torch.load(args.finetuned_path_action, map_location='cpu')
        action_model.load_state_dict(finetuned_ckpt)

    action_model.cuda()
    args.train_tower = 'event'

    event_model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=None, task_config=args)

    if args.finetuned_path is not None:
        finetuned_ckpt = torch.load(args.finetuned_path, map_location='cpu')
        event_model.load_state_dict(finetuned_ckpt)

    event_model.cuda()


    return object_model,action_model,event_model





def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list,action):
    if model.task_config.train_tower=='event':
        sim_matrix = []
        new_visual_embed = []
        for idx1, b1 in enumerate(batch_list_t):

            input_mask, *_tmp = b1
            # print('input mask')
            sequence_output = batch_sequence_output_list[idx1]
            each_row = []
            for idx2, b2 in enumerate(batch_list_v):
                video_mask, *_tmp = b2
                visual_output = batch_visual_output_list[idx2]

                b1b2_logits, cur_visual= model.get_similarity_logits(sequence_output, visual_output, video_mask,action=action)
                b1b2_logits = b1b2_logits.cpu().detach().numpy()
                each_row.append(b1b2_logits)
                if idx1==0:
                    new_visual_embed.append(cur_visual)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row) 

        return sim_matrix,new_visual_embed
    else:
        sim_matrix = []
        for idx1, b1 in enumerate(batch_list_t):
            # if idx1!=0:
            #     print('sf')
            input_mask, *_tmp = b1
            # print('input mask')
            sequence_output = batch_sequence_output_list[idx1]
            each_row = []
            for idx2, b2 in enumerate(batch_list_v):
                video_mask, *_tmp = b2
                visual_output = batch_visual_output_list[idx2]

                b1b2_logits= model.get_similarity_logits(sequence_output, visual_output, video_mask,action=action)
                b1b2_logits = b1b2_logits.cpu().detach().numpy()
                each_row.append(b1b2_logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row) 
        return sim_matrix

def _run_on_single_gpu_event_model(model,batch_input_ids,batch_sequence_obj,batch_sequence_act):
    all_weight_act,all_weight_obj = [],[]
    for index,input_ids in enumerate(batch_input_ids):
        each_row_act = []
        each_row_obj = []
        for index2,sequence_obj in enumerate(batch_sequence_obj):
            weight_action,weight_object = model.get_weight_action(input_ids, batch_sequence_act[index2])
            each_row_act.append(weight_action.cpu().detach().numpy())
            each_row_obj.append(weight_object.cpu().detach().numpy())
        each_row_act = np.concatenate(tuple(each_row_act),axis=-1)
        each_row_obj = np.concatenate(tuple(each_row_obj),axis=-1)
        all_weight_act.append(each_row_act)
        all_weight_obj.append(each_row_obj)
    return all_weight_obj,all_weight_act
    
def eval_epoch(args, model, test_dataloader, device, n_gpu,logger=None,object_model=None,action_model=None):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        
        batch_list_v_obj = []
        batch_list_v_act = []
        batch_sequence_output_list_obj = []
        batch_sequence_output_list_act = []
        batch_visual_output_list_obj = []
        batch_visual_output_list_act = []
        total_video_num = 0

        batch_list_ids = []
        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            if args.train_tower!='object':
                if args.action_model=='wham':
                    input_ids, attention_mask,video, video_mask,key_points,init_kp,key_mask,motion_mask = batch
                elif args.action_model =='motionbert':
                    input_ids, attention_mask,video, video_mask,key_points,motion_mask = batch
                    init_kp = key_mask= None
                else:
                    raise ValueError()
            else:
                input_ids,attention_mask, video, video_mask = batch
                key_points = init_kp = key_mask = motion_mask = None
            # import ipdb;ipdb.set_trace() input_ids, attention_mask:  [bsz,1,77]
            if args.train_tower=='object':
                batch_list_v.append((video_mask,))
                sequence_output, visual_output = model.get_sequence_visual_output(input_ids,video, video_mask)
            elif args.train_tower == 'action':
                sequence_output, visual_output = model.get_verb_action_output(input_ids,attention_mask,key_points,init_kp,key_mask)
                batch_list_v.append((motion_mask,))
            else:
                batch_list_ids.append(input_ids.squeeze(1))
                batch_list_v_obj.append((video_mask,))
                batch_list_v_act.append((motion_mask,))
                sequence_output_obj, visual_output_obj = object_model.get_sequence_visual_output(input_ids,video, video_mask)
                sequence_output_act, visual_output_act = action_model.get_verb_action_output(input_ids,attention_mask,key_points,init_kp,key_mask)
                batch_sequence_output_list_obj.append(sequence_output_obj)
                batch_sequence_output_list_act.append(sequence_output_act)
                batch_visual_output_list_obj.append(visual_output_obj)
                batch_visual_output_list_act.append(visual_output_act)
                batch_list_t.append((attention_mask,))
                print("{}/{}\r".format(bid, len(test_dataloader)), end="")
                continue
            batch_sequence_output_list.append(sequence_output)
            batch_list_t.append((attention_mask,))

            batch_visual_output_list.append(visual_output)



            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------

        if args.train_tower=='object':
            sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list,False)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        elif args.train_tower =='action':
            sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list,True)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        else:
            object_model.task_config.train_tower = 'event'
            sim_matrix_obj,batch_new_visual_obj = _run_on_single_gpu(object_model, batch_list_t, batch_list_v_obj, batch_sequence_output_list_obj, batch_visual_output_list_obj,False)
            sim_matrix_obj = np.concatenate(tuple(sim_matrix_obj), axis=0)
            sim_matrix_action,batch_new_visual_act = _run_on_single_gpu(action_model, batch_list_t, batch_list_v_act, batch_sequence_output_list_act, batch_visual_output_list_act,True)
            sim_matrix_action = np.concatenate(tuple(sim_matrix_action), axis=0)
            weight_object,weight_action = _run_on_single_gpu_event_model(model,batch_list_ids,batch_new_visual_obj,batch_new_visual_act)
            weight_object = np.concatenate(tuple(weight_object), axis=0)
            weight_action = np.concatenate(tuple(weight_action), axis=0)            
            sim_matrix = sim_matrix_action * weight_action + sim_matrix_obj * weight_object
        # sim_matrix_dsl = sim_matrix * softmax(sim_matrix, axis=0)
        # sim_matrix_dsl_T = sim_matrix.T * softmax(sim_matrix.T, axis=0)


    logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
    # explain(args,sim_matrix)
    # exit()
    # import pdb;pdb.set_trace()
    tv_metrics = compute_metrics(sim_matrix)
    vt_metrics = compute_metrics(sim_matrix.T)
    # dsl_tv_metrics = compute_metrics(sim_matrix_dsl)
    # dsl_vt_metrics = compute_metrics(sim_matrix_dsl.T)
    # dsl_vt_metrics = compute_metrics(sim_matrix_dsl_T)
    logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
        
    # dsl output
    logger.info("------------------------------------------------------------")
    # logger.info("DSL Text-to-Video:")
    # logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f}  - Median R: {:.1f} - Mean R: {:.1f}'.
    #             format(dsl_tv_metrics['R1'], dsl_tv_metrics['R5'], dsl_tv_metrics['R10'], dsl_tv_metrics['MR'], dsl_tv_metrics['MeanR']))
    # logger.info("DSL Video-to-Text:")
    # logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f}  - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
    #             format(dsl_vt_metrics['R1'], dsl_vt_metrics['R5'], dsl_vt_metrics['R10'],  dsl_vt_metrics['MR'], dsl_vt_metrics['MeanR']))

    # logger.info("------------------------------------------------------------")
    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f}  - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'],  tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'],  vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    return R1


def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.rank)

    if args.verb_model=='bert':
        tokenizer = AutoTokenizer.from_pretrained("llms/bert-base-uncased")
    elif args.verb_model == 'internvideo':
        tokenizer = ClipTokenizer()
    else:
        raise ValueError()

    # assert  args.task_type == "retrieval"
    if args.train_tower=='event':
        object_model,action_model,model = init_model_for_three(args, device, n_gpu, args.rank)
        # model = None
    else:
        object_model,action_model = None,None
        model = init_model(args, device, n_gpu, args.rank)

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    ## ####################################
    ### test ####
    #######################################
    if args.rank == 0:
        eval_epoch(args, model, test_dataloader, device, n_gpu,logger,object_model,action_model)

import json
import shutil
def explain(args,x):


    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    # ind = np.where(ind == 0)
    # print(np.where(ind == 0)[1])
    # ind = ind[1]
    ind = np.argmax(ind == 0, axis=1)
    # print(ind)

    
    json_path = args.val_csv
    right_json = 'dataset/explain/explain_right_action_model.json'
    # wrong_json = 'dataset/motion-x++-videos/explain_wrong_vid.json'
    
    with open(json_path,'r') as f:
        files = json.load(f)
    # import pdb;pdb.set_trace()
    right = []
    wrong = []
    for i,item in enumerate(tqdm(ind.tolist())):
        temp_dict = {}
        if item==0:
            try:
                temp_dict['text'] = files[i]['text']
            except:
                import pdb;pdb.set_trace()
            temp_dict['video_path'] = files[i]['video_path']
            right.append(temp_dict)
        else:
            temp_dict['text'] = files[i]['text']
            temp_dict['predict_rank'] = item
            temp_dict['gt_video_path'] = files[i]['video_path']
            cur_index = x[i].tolist().index(-sx[i][0])
            # print(cur_index)
            temp_dict['pre_video_path'] = files[cur_index]['video_path']
            temp_dict['text2'] = files[cur_index]['text']
            wrong.append(temp_dict)
    
    with open(right_json,'w') as f:
        json.dump(right,f)
    exit()
    with open(wrong_json,'w') as f:
        json.dump(wrong,f)
    
    wrong_sample_path = 'dataset/motion-x++-videos/wrong_samples'
    if os.path.exists(wrong_sample_path):
        shutil.rmtree(wrong_sample_path)
    
    os.makedirs(wrong_sample_path)
    for id,item in enumerate(wrong):
        cur_path = os.path.join(wrong_sample_path,str(id))
        os.makedirs(cur_path,exist_ok=True)
        shutil.copyfile(item['gt_video_path'], os.path.join(cur_path,'gt_video.mp4'))
        shutil.copyfile(item['pre_video_path'],os.path.join(cur_path,'pre_video.mp4'))
        info = 'gt text: '+item['text']+'\n' + 'predict rank: '+str(item['predict_rank'])+'\n' + 'text of predict video: '+item['text2']
        with open(os.path.join(cur_path,'text.txt'),'w') as f:
            f.write(info)



    
if __name__ == "__main__":
    main()