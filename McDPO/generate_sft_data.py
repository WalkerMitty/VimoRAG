import json
import os
from tqdm import tqdm
import numpy as np
PROMPT = 'Generate a sequence of motion tokens matching the following human motion description. You can use the video as a reference. Video information: <video>\n Motion description: {}'


raw_text_path = 'MotionGPT/dataset/t2m/texts'
motiongpt_path = 'generation/MotionGPT/data'
sft_data = 'dataset/sft_data'

retrieval_top1 = 'dataset/retrieval_inference_wild/train_t2m_top1_wild_new.json'
def read_text(text_path):
    captions = []
    with open(text_path,'r') as f:
        for line in f.readlines():
            # print(line)
            line_split = line.strip().split('#')
            caption = line_split[0]
            try:
                tokens = line_split[1].split(' ')
            except:
                continue
            f_tag = float(line_split[2])
            to_tag = float(line_split[3])
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag
            captions.append(caption)
    return captions
def generation(split):

    motion_split_path = os.path.join(motiongpt_path,split+'_t2m.json') 
    with open(motion_split_path,'r') as f:
        motion_split_data = json.load(f)
    with open(retrieval_top1,'r') as f:
        retrieval_data = json.load(f)

    final_data = []
    max = 0
    ids_set = set()
    input_set = set()
    for item in tqdm(motion_split_data):
        # input = item['input']

        output = item['output']
        # print(output)
        # exit()
        # if len(output.split(','))>max:
        #     max = len(output.split(','))
        id = item['motion']
        text_path = os.path.join(raw_text_path,id+'.txt')
        inputs = read_text(text_path)
        # inputs = [input]
        # video_path = os.path.join(HumanML3D_videos,id+'.mp4')
        if id not in ids_set:
            ids_set.add(id)
            for input in inputs:
                if input in input_set:
                    continue
                input_set.add(input)
                temp_dict = {}
                temp_dict['source'] = 't2m'
                temp_dict['type'] = 'conv'
                temp_dict['id'] = id
                # temp_dict['video'] = id+'.mp4' #ground truth in t2m video format
                temp_dict['video'] = retrieval_data[input]
                # temp_dict['video'] = ''
                temp_dict['instruction'] = None
                h_value = PROMPT.format(input)
                # h_value = PROMPT_TEXT_ONLY.format(input)
                temp_dict['conversations'] = [{'from':'human','value':h_value},{'from':'gpt','value':output}]
                final_data.append(temp_dict)
    # print(final_data)

    
    with open(os.path.join(sft_data,split+'_top1_kit_new.json'),'w') as f:
        json.dump(final_data,f)

def generation_pretrain():
    final_data = []
    all_anno_file = 'dataset/wild_motion_videos/mergev1_caption.json' 
    with open(all_anno_file,'r') as f:
        all_file = json.load(f)
    
    for item in tqdm(all_file):
        temp_dict = {}
        temp_dict['source'] = 'mergev1'
        temp_dict['type'] = 'conv'
        temp_dict['instruction'] = None
        for key,value in item.items():
            temp_dict['video'] = key
            temp_dict['conversations'] = [{'from':'human','value':PROMPT_PRETRAIN},{'from':'gpt','value':value}]
            final_data.append(temp_dict)
            break


    with open(os.path.join(sft_data,'mergev1_pretrain.json'),'w') as f:
        json.dump(final_data,f)
    



if __name__ == '__main__':
    generation('train')
