import argparse
import torch
import json
import os
from tqdm import tqdm
def get_wild()->list:
    video_data_base = '../resources/motionbertprocessed.pth'
    raw_data = torch.load(video_data_base)
    all_videos_path = list(raw_data.keys())
    all_videos_path = [item for item in all_videos_path if "haa500" in item]
    return all_videos_path
def construct_wild(args):
    if not os.path.exists("../output"):
        os.makedirs("../output")
    wild_videos_database = get_wild()
    all_text = [args.text.strip()]
    print('the query number is ',len(all_text))
    # exit()
    final_data = []
    for index,item in enumerate(tqdm(wild_videos_database)):
        temp_dict = {}
        temp_dict['video_path'] = item
        if index<len(all_text):
            temp_dict['text'] = all_text[index]
        else:
            temp_dict['text'] = 'padding'
        final_data.append(temp_dict)
    with open(args.demo_file,'w') as f:
        json.dump(final_data,f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Input your instruction here")
    parser.add_argument("--demo_file", type=str, default="../output/demo_input.json",help="Intermediate file for retrieval")
    args = parser.parse_args()
    construct_wild(args)