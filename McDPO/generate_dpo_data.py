
import json
from tqdm import tqdm
import copy
def format_dpo_file():

    sft_json = 'dataset/sft_data/train_random_t2m.json'
    selected_json = 'dataset/dpo_selection/no_motion_r128_a256_bsz8x8_epoch3_random_train_3seed.json'
    dpo_json = 'dataset/dpo_selection/no_motion_r128_a256_bsz8x8_epoch3_random_train_3seed_self.json'

    
    with open(sft_json,'r') as f:
        sft_data = json.load(f)
    with open(selected_json,'r') as f:
        selected_data = json.load(f)
    # with open(gt_json,'r') as f:
    #     gt_data = json.load(f)
    final_data = []
    #selected_data: key:value key is input_text, value is the predicted motions
    for item in tqdm(sft_data):
        temp_dict = {}
        cur_text = item['conversations'][0]['value'].split('Motion description: ')[1]
        conv = item['conversations']
        try:
            chosen_answer = selected_data[cur_text]['chosen']
            # chosen_answer = gt_data[cur_text]
        except:
            continue
        rejected_answer = selected_data[cur_text]['rejected']
        temp_dict.update(item)
        '''
        conv[1]['value'] = chosen_answer
        temp_dict['chosen_conversations'] = conv
        conv[1]['value'] = rejected_answer
        temp_dict['rejected_conversations'] = conv
        '''
        chosen_conv = copy.deepcopy(conv)
        chosen_conv[1]['value'] = chosen_answer
        temp_dict['chosen_conversations'] = chosen_conv

        rejected_conv = copy.deepcopy(conv)
        rejected_conv[1]['value'] = rejected_answer
        temp_dict['rejected_conversations'] = rejected_conv
        final_data.append(temp_dict)
    print(len(final_data))
    with open(dpo_json,'w') as f:
        json.dump(final_data,f)


if __name__ == '__main__':
    format_dpo_file()
