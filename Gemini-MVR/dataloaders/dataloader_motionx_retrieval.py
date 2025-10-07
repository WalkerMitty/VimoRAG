from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import io
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
# from dataloaders.rawvideo_util import RawVideoExtractor
import json

    
from decord import VideoReader, cpu
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
# verb_model = 'bert'

def closest_multiple_of_bsz(n,bsz):
    closest_multiple = (n // bsz) * bsz
    
    return int(closest_multiple)
class MOTIONX_DataLoader(Dataset):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            json_path,
            tokenizer,
            max_words=30,
            feature_framerate=1,
            max_frames=100,
            min_frames = 8,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            num=-1,
            threads=24,
            bsz = 128,
            verb_model = '',
            train_tower = '',
            action_model = '',
    ):
        if train_tower!='object':
            if action_model=='wham':
                self.cache_keypoints = torch.load('../cache_keypoints/cache_keypoints_mergev1.pth')
            elif action_model=='motionbert':
                self.cache_keypoints = torch.load('../resources/motionbertprocessed.pth')
            else:
                raise ValueError()
        self.action_model = action_model
        self.train_tower = train_tower
        self.verb_model = verb_model
        with open(json_path,'r') as f:
            files = json.load(f)
        if num!=-1:

            gap = closest_multiple_of_bsz(len(files)/threads,bsz)
            start = num*gap
            end = (num+1)*gap
            if num==threads-1:
                files = files[start:]
            else:
                files = files[start:end]

        video_ids = [item['video_path'].replace('/workspace/','/data/nas/') for item in files]
        sents = [item['text'] for item in files]
        self.data = {}
        if 'text_id' in files[0]:
            text_ids =  [item['text_id'] for item in files]
            self.data['text_id'] = text_ids

        self.data['video_id'] = video_ids
        self.data['sentence'] = sents
        # self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.image_resolution = image_resolution
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        # self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.transform = Compose([
                    Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(image_resolution),
                    lambda image: image.convert("RGB"),
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    def __len__(self):
        return len(self.data['sentence'])

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        # pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            # pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask,choice_video_ids
    def _get_text_bert(self,video_id,sentence):
        choice_video_ids = [video_id]
        sentence = [sentence]
        inputs = self.tokenizer(sentence, return_tensors="np", padding=True, truncation=True, max_length=self.max_words)
        return inputs['input_ids'], inputs['attention_mask'],choice_video_ids
    def _get_rawvideo_dec(self, choice_video_ids, s=None, e=None):
        # speed up video decode via decord.
        # video_mask = np.zeros(self.max_frames, dtype=np.long)
        min_frames = self.min_frames
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        
        # max_video_length = 0
        max_video_length = [0] * len(choice_video_ids)

        # T x 3 x H x W
        # video = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float)
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.image_resolution, self.image_resolution), dtype=np.float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1
        # video_path = self.video_dict[video_id]
        for i, video_id in enumerate(choice_video_ids):
    
            # video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            video_path = video_id

            vreader = VideoReader(video_path, ctx=cpu(0))
        
            fps = vreader.get_avg_fps()
            f_start = 0 if start_time is None else int(start_time * fps)
            f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
            num_frames = f_end - f_start + 1
            if num_frames > 0:
                # T x 3 x H x W
                # sample_fps = int(self.video_framerate)
                sample_fps = int(self.feature_framerate)
                t_stride = int(round(float(fps) / sample_fps))

                all_pos = list(range(f_start, f_end + 1, t_stride))
                if len(all_pos) > self.max_frames:
                    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=self.max_frames, dtype=int)]
                elif len(all_pos) < min_frames:
                    if num_frames < min_frames:
                        min_frames = num_frames
                    t_stride = max(1, (f_end - f_start) // (min_frames - 1))
                    adjusted_f_end = f_start + t_stride * (min_frames - 1)
                    sample_pos = list(range(f_start, adjusted_f_end + 1, t_stride))
                else:
                    sample_pos = all_pos

                patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
                patch_images = torch.stack([self.transform(img) for img in patch_images])
                
                patch_images = patch_images.unsqueeze(1)
                
                slice_len = patch_images.shape[0]
                # max_video_length = max_video_length if max_video_length > slice_len else slice_len
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = patch_images
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        # video_mask[:max_video_length] = [1] * max_video_length
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        #print(video.shape, video_mask.shape)
        return video, video_mask
    def _get_motion(self,video_paths):
        # if self.cache_keypoints=={}:
        #     self.get_cache_keypoints()
        all_keypoints = []
        all_init_kp = []
        all_key_mask = []
        motion_mask = np.zeros((len(video_paths), self.max_frames), dtype=np.long)
        for i,video_file_path in enumerate(video_paths):
            try:
                cur_keypoints = self.cache_keypoints[video_file_path]
            except:
                print('cache_keypoints read fails, loading the default one')
                cur_keypoints = self.cache_keypoints['../resources/wild_motion_videos/aslan/7-0005.mp4']
            keypoints = cur_keypoints['keypoints'] #[1,16,37]
            init_kp = cur_keypoints['init_kp'] #[1,1,88]
            key_mask = cur_keypoints['key_mask'] # [1,16,17]这个key mask不是长度的padding mask，是一种特征
            motion_mask[i][:key_mask.shape[1]] = [1] * key_mask.shape[1]
            if key_mask.size()[1]< self.max_frames: #batch 训练进行padding
                padding_num = self.max_frames - key_mask.size()[1]
                temp = torch.zeros(1,padding_num, 17, dtype=torch.bool)
                temp2 = torch.zeros(1,padding_num, 37,dtype=keypoints.dtype)
                # import ipdb;ipdb.set_trace()
                key_mask = torch.cat((key_mask,temp),dim=1)
                keypoints = torch.cat((keypoints,temp2),dim=1)

            #进行padding！
            all_keypoints.append(keypoints.squeeze(0).numpy())
            all_init_kp.append(init_kp.squeeze(0).numpy())
            all_key_mask.append(key_mask.squeeze(0).numpy())
            
        #return all_keypoints [len(video_paths), 16,37]
        return np.array(all_keypoints),np.array(all_init_kp),np.array(all_key_mask), motion_mask


    def _get_motion_motionbert(self,video_paths):
        all_keypoints = []
        motion_mask = np.zeros((len(video_paths), self.max_frames), dtype=np.long)
        for i,video_file_path in enumerate(video_paths):
            try:
                cur_keypoints = self.cache_keypoints[video_file_path] #[n,17,3]
            except:
                print('cache_keypoints read fails, loading the default one')
                cur_keypoints = self.cache_keypoints['../resources/wild_motion_videos/aslan/7-0005.mp4']
            
            motion_mask[i][:cur_keypoints.shape[0]] = [1] * cur_keypoints.shape[0]
            if cur_keypoints.shape[0]< self.max_frames:
                padding_num = self.max_frames - cur_keypoints.shape[0]
                temp = np.zeros((padding_num, 17, 3), dtype=cur_keypoints.dtype)
                cur_keypoints = np.concatenate((cur_keypoints, temp), axis=0)

            all_keypoints.append(cur_keypoints)
            
        return np.array(all_keypoints),motion_mask

    def __getitem__(self, idx):
        video_id = self.data['video_id'][idx] 
        sentence = self.data['sentence'][idx]
        if self.verb_model=='bert':
            pairs_text,pairs_mask,choice_video_ids = self._get_text_bert(video_id,sentence)
        elif self.verb_model=='internvideo':
            pairs_text, pairs_mask, choice_video_ids = self._get_text(video_id, sentence)
        else:
            raise ValueError()
        video, video_mask = self._get_rawvideo_dec(choice_video_ids)
        # if 'text_id' in self.data: #inference mode
        #     return pairs_text, pairs_mask, pairs_segment, video, video_mask, self.data['text_id'][idx],video_id
        if self.train_tower!='object':
            if self.action_model =='wham':
                key_points,init_kp,key_mask,motion_mask = self._get_motion(choice_video_ids)
                return pairs_text, pairs_mask, video, video_mask,key_points,init_kp,key_mask,motion_mask
            elif self.action_model == 'motionbert':
                key_points,motion_mask = self._get_motion_motionbert(choice_video_ids)
                return pairs_text, pairs_mask, video, video_mask,key_points,motion_mask
            else:
                raise ValueError()
        else:
            return pairs_text, pairs_mask, video, video_mask

class MOTIONX_TrainDataLoader(Dataset):
    """motionx train dataset loader."""
    def __init__(
            self,
            json_path,
            tokenizer,
            max_words=30,
            feature_framerate=1,
            max_frames=100,
            min_frames = 8,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            verb_model = '',
            train_tower = '',
            action_model = '',

    ):
        # self.csv = pd.read_csv(csv_path)
        self.train_tower = train_tower
        self.action_model = action_model
        self.data = json.load(open(json_path, 'r'))
        # size = int(len(self.data)/10)
        # self.data = self.data[:size]
        
        # self.features_path = features_path
        self.verb_model = verb_model
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.image_resolution = image_resolution
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        # self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        # if self.unfold_sentences:
        # train_video_ids = list(self.csv['video_id'].values)
        self.sentences_dict = {}
        for itm in self.data:
            self.sentences_dict[len(self.sentences_dict)] = (itm['video_path'].replace('/workspace/','/data/nas/'), itm['text'])
        self.sample_len = len(self.sentences_dict)
        # self.cache_keypoints = {}
        if self.train_tower!='object':
            if action_model=='wham':
                self.cache_keypoints = torch.load('../cache_keypoints/cache_keypoints_mergev1.pth')
            elif action_model=='motionbert':
                self.cache_keypoints = torch.load('../resources/motionbertprocessed.pth')
            else:
                raise ValueError()

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = Compose([
                    Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(image_resolution),
                    lambda image: image.convert("RGB"),
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        # pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            # pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask,choice_video_ids
    def _get_text_bert(self,video_id,sentence):
        choice_video_ids = [video_id]
        sentence = [sentence]
        inputs = self.tokenizer(sentence, return_tensors="np", padding=True, truncation=True, max_length=self.max_words)
        return inputs['input_ids'], inputs['attention_mask'],choice_video_ids
    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words


    def _get_rawvideo_dec(self, choice_video_ids, s=None, e=None):
        # speed up video decode via decord.
        # video_mask = np.zeros(self.max_frames, dtype=np.long)
        min_frames = self.min_frames
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        
        # max_video_length = 0
        max_video_length = [0] * len(choice_video_ids)

        # T x 3 x H x W
        # video = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float)
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.image_resolution, self.image_resolution), dtype=np.float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1
        # video_path = self.video_dict[video_id]
        for i, video_id in enumerate(choice_video_ids):
            video_path = video_id

            vreader = VideoReader(video_path, ctx=cpu(0))
        
            fps = vreader.get_avg_fps()
            f_start = 0 if start_time is None else int(start_time * fps)
            f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
            num_frames = f_end - f_start + 1
            if num_frames > 0:
                # T x 3 x H x W
                # sample_fps = int(self.video_framerate)
                sample_fps = int(self.feature_framerate)
                t_stride = int(round(float(fps) / sample_fps))

                all_pos = list(range(f_start, f_end + 1, t_stride))
                if len(all_pos) > self.max_frames:
                    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=self.max_frames, dtype=int)]
                elif len(all_pos) < min_frames:
                    if num_frames < min_frames:
                        min_frames = num_frames
                    t_stride = max(1, (f_end - f_start) // (min_frames - 1))
                    adjusted_f_end = f_start + t_stride * (min_frames - 1)
                    sample_pos = list(range(f_start, adjusted_f_end + 1, t_stride))
                else:
                    sample_pos = all_pos

                patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
                patch_images = torch.stack([self.transform(img) for img in patch_images])
                
                patch_images = patch_images.unsqueeze(1)
                
                slice_len = patch_images.shape[0]
                # max_video_length = max_video_length if max_video_length > slice_len else slice_len
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = patch_images
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        # video_mask[:max_video_length] = [1] * max_video_length
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        #print(video.shape, video_mask.shape)
        return video, video_mask
    def _get_motion(self,video_paths):
        # if self.cache_keypoints=={}:
        #     self.get_cache_keypoints()
        all_keypoints = []
        all_init_kp = []
        all_key_mask = []
        motion_mask = np.zeros((len(video_paths), self.max_frames), dtype=np.long)
        for i,video_file_path in enumerate(video_paths):
            try:
                cur_keypoints = self.cache_keypoints[video_file_path]
            except:
                print('cache_keypoints read fails, loading the default one')
                cur_keypoints = self.cache_keypoints['../resources/wild_motion_videos/aslan/7-0005.mp4']
            keypoints = cur_keypoints['keypoints'] #[1,16,37]
            init_kp = cur_keypoints['init_kp'] #[1,1,88]
            key_mask = cur_keypoints['key_mask'] # [1,16,17]
            motion_mask[i][:key_mask.shape[1]] = [1] * key_mask.shape[1]
            if key_mask.size()[1]< self.max_frames: #batch 
                padding_num = self.max_frames - key_mask.size()[1]
                temp = torch.zeros(1,padding_num, 17, dtype=torch.bool)
                temp2 = torch.zeros(1,padding_num, 37,dtype=keypoints.dtype)
                # import ipdb;ipdb.set_trace()
                key_mask = torch.cat((key_mask,temp),dim=1)
                keypoints = torch.cat((keypoints,temp2),dim=1)


            all_keypoints.append(keypoints.squeeze(0).numpy())
            all_init_kp.append(init_kp.squeeze(0).numpy())
            all_key_mask.append(key_mask.squeeze(0).numpy())

        #return all_keypoints [len(video_paths), 16,37]
        return np.array(all_keypoints),np.array(all_init_kp),np.array(all_key_mask), motion_mask
    def _get_motion_motionbert(self,video_paths):
        all_keypoints = []
        motion_mask = np.zeros((len(video_paths), self.max_frames), dtype=np.long)
        for i,video_file_path in enumerate(video_paths):
            try:
                cur_keypoints = self.cache_keypoints[video_file_path] #[n,17,3]
            except:
                print('cache_keypoints read fails, loading the default one')
                cur_keypoints = self.cache_keypoints['../resources/wild_motion_videos/aslan/7-0005.mp4']
            
            motion_mask[i][:cur_keypoints.shape[0]] = [1] * cur_keypoints.shape[0]
            if cur_keypoints.shape[0]< self.max_frames:
                padding_num = self.max_frames - cur_keypoints.shape[0]
                temp = np.zeros((padding_num, 17, 3), dtype=cur_keypoints.dtype)
                cur_keypoints = np.concatenate((cur_keypoints, temp), axis=0)


            all_keypoints.append(cur_keypoints)
            
        #return all_keypoints [len(video_paths), 16,17,3]
        return np.array(all_keypoints),motion_mask
    def __getitem__(self, idx):

        video_id, caption = self.sentences_dict[idx]
        if self.verb_model == 'bert':
            pairs_text,attention_mask,choice_video_ids = self._get_text_bert(video_id,caption)
        elif self.verb_model == 'internvideo':
            pairs_text, attention_mask,choice_video_ids = self._get_text(video_id, caption)
        else:
            raise ValueError()
        # video, video_mask = self._get_rawvideo(choice_video_ids)
        video, video_mask = self._get_rawvideo_dec(choice_video_ids)

        if self.train_tower!='object':
            if self.action_model =='wham':
                key_points,init_kp,key_mask,motion_mask = self._get_motion(choice_video_ids)
                return pairs_text, attention_mask, video, video_mask,key_points,init_kp,key_mask,motion_mask
            elif self.action_model == 'motionbert':
                key_points,motion_mask = self._get_motion_motionbert(choice_video_ids)
                return pairs_text, attention_mask, video, video_mask,key_points,motion_mask
            else:
                raise ValueError()
        else:
            return pairs_text,attention_mask, video, video_mask




