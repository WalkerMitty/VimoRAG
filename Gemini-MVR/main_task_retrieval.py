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
from transformers import AutoTokenizer
from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT
import torch.distributed as dist
import subprocess
import torch.nn as nn
from scipy.special import softmax
from tqdm import tqdm
# torch.distributed.init_process_group(backend="nccl")
from inference import _run_on_single_gpu,eval_epoch
global logger

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    # parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.") remove
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/MSR-VTT/anns/MSRVTT_train.9k.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/MSR-VTT/anns/MSRVTT_JSFUSION_test.csv', help='')
    # parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path') remove
    # parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    # parser.add_argument('--features_path', type=str, default='s3://video_pub/MSR-VTT/videos', help='feature path') remove

    parser.add_argument('--num_thread_reader', type=int, default=4, help='num workers in dataloader')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate') #only be used in ``object`` tower
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
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default='/path/to/save/your/experiments/', type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    # parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.") #remove this one
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model (only optimizer).")
    parser.add_argument("--resume_model_true", default=None, type=str, required=False, help="Resume train model.")
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

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.') # only used in object model
    # parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    # parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    # parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.") #remove this param
    # parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="") remove

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
    #                     choices=["meanP"],
    #                     help="choice a similarity header.")




    ## hierarchical model
    parser.add_argument("--mlp_lr", type=float, default=1e-3, help="the lr of action model")
    parser.add_argument("--train_tower", type=str, choices = ['action','object','event'],help="the training stage of the hierarchical model, 'action' means only action model is trained, 'event' means freeze the other models, keeping the event extractor tuning ")
    parser.add_argument("--verb_model", type=str, choices=['bert','internvideo'], help="the type of text encoder in verb model")
    parser.add_argument("--action_model", type=str, choices=['wham','motionbert'], help="the type of action encoder")
    parser.add_argument("--resume_model_true_object", default=None, type=str, required=False, help="saved object model for event model")
    parser.add_argument("--resume_model_true_action", default=None, type=str, required=False, help="pretrained action model for action model")
    #### CLIP EVL ######
    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="ViT Base or Large")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Give the upstream pretrained checkpoint (not the finetuned one)")
    parser.add_argument("--clip_evl", action='store_true', help="whether to activate clip_evl")
    parser.add_argument("--mergeclip", type=bool, default=False, help="whether to merge clip weight")
    parser.add_argument("--mergeweight", type=float, default=0.5, help="merge weight from 0 to 1")


    ### DRL ###
    # parser.add_argument("--interaction", type=str, default='no', help="interaction mode, refer to DRL") remove
    # parser.add_argument("--wti_arch", type=int, default=0, help="wti architecture, refer to DRL") remove
    # parser.add_argument("--cdcr", type=int, default=0, help="which cdcr type, refer to DRL") remove
    
    args = parser.parse_args()


    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

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
        print('start first')
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
        master_port = os.environ.get('MASTER_PORT', '29491')
        ## manually set is also ok ##
        master_port = "29411"
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
    if args.resume_model_true:
        model.load_state_dict(torch.load(args.resume_model_true)) 
    # import ipdb;ipdb.set_trace()
    # model.to(device)
    model.cuda()

    return model

def init_model_for_three(args, device, n_gpu, local_rank):



    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    args.train_tower = 'object'
    object_model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=None, task_config=args)

    if args.resume_model_true_object is not None:
        finetuned_ckpt = torch.load(args.resume_model_true_object, map_location='cpu')
        object_model.load_state_dict(finetuned_ckpt)

    object_model.cuda()
    for p in object_model.parameters():
        p.requires_grad= False
    args.train_tower = 'action'
    action_model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=None, task_config=args)

    if args.resume_model_true_action is not None:
        finetuned_ckpt = torch.load(args.resume_model_true_action, map_location='cpu')
        action_model.load_state_dict(finetuned_ckpt)

    action_model.cuda()


    for p in action_model.parameters():
        p.requires_grad= False

    args.train_tower = 'event'

    event_model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=None, task_config=args)

    if args.resume_model_true is not None:
        finetuned_ckpt = torch.load(args.resume_model_true, map_location='cpu')
        event_model.load_state_dict(finetuned_ckpt)

    event_model.cuda()


    return object_model,action_model,event_model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #video_weight_fc
    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)] 
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)] 

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n] 

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2


    if args.train_tower=='action': 
        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay,'lr':args.mlp_lr},
            {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0, 'lr':args.mlp_lr}
        ]
    elif args.train_tower=='object':
        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
            {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
        ]
    else:
        event_model_p = []
        for n,p in model.named_parameters():
            if 'event_extractor' not in n:
                p.requires_grad = False
            else:
                event_model_p.append(p)

        optimizer_grouped_parameters = [
            {'params': event_model_p, 'weight_decay': weight_decay, 'lr': args.mlp_lr},
        ]
    # import ipdb;ipdb.set_trace()
    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(),
                                                          find_unused_parameters=False)


    return optimizer, scheduler, model

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin")
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin")
    
    torch.save(model_to_save.state_dict(), output_model_file)
    
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': tr_loss,
        }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
       model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0,objectmodel=None,actionmodel=None):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(tqdm(train_dataloader)):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
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


        loss = model(input_ids,attention_mask,video, video_mask,key_points,init_kp,key_mask,motion_mask,objectmodel,actionmodel)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
            # loss_action = loss_action.mean()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if args.train_tower == 'object':
                if hasattr(model, 'module'):
                    torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
                else:
                    torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and args.rank == 0:
                print('lr: ',set(optimizer.get_lr())) 
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f,  Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss), 
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step



def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.rank)
    if args.verb_model=='bert':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif args.verb_model == 'internvideo':
        tokenizer = ClipTokenizer()
    else:
        raise ValueError()
    # import pdb;pdb.set_trace()
    # assert  args.task_type == "retrieval"
    if args.train_tower=='event':
        objectmodel,actionmodel,model = init_model_for_three(args,device,n_gpu,args.rank)
    else:
        model = init_model(args, device, n_gpu, args.rank)
        objectmodel,actionmodel=None,None
    ## ####################################
    # freeze testing
    ## ####################################
    if args.train_tower =='object':
        assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
        if hasattr(model, "clip") and args.freeze_layer_num > -1:
            for name, param in model.clip.named_parameters():
                # top layers always need to train
                if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                        or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                    continue    # need to train
                elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                    layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                    if layer_num >= args.freeze_layer_num:
                        continue    # need to train
                
                if name.find('.dpe.') >=0 or name.find('.dec.') >= 0:
                    continue # need to train
                
                if args.linear_patch == "3d" and name.find("conv2."):
                    continue # need to train
                else:
                    # paramenters which < freeze_layer_num will be freezed
                    param.requires_grad = False
                    print('freezing: ', name, name.find('dpe.'), name.find('dec.'), param.shape)

    
    # exit(0)

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
    # train and eval
    ## ####################################
    # if args.rank == 0:
    
    # exit()
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.rank, coef_lr=coef_lr)
        # eval_epoch(args, model, test_dataloader, device, n_gpu,logger)
        if args.rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001

        best_output_model_file = "None"
        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
            resumed_loss = checkpoint['loss']
        print('begin training!!!!!!!!')
        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch) 
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.rank,objectmodel=objectmodel,actionmodel=actionmodel)
            if args.rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                ########## here we save the last ckpt #############
                ########## feel free to modify this for saving the best ckpt #######
                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

    elif args.do_eval:
        if args.rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu,logger)

if __name__ == "__main__":
    main()