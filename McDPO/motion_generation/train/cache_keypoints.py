import json
from videogpt_plus.constants import *
from videogpt_plus.model.wham_dataloader.detector import DetectionModel
from videogpt_plus.model.wham_dataloader.extractor import FeatureExtractor
from videogpt_plus.model.wham_dataloader.normalizer import Normalizer
from videogpt_plus.model.wham_dataloader.smpl import SMPL
import sys
from videogpt_plus.model.dataloader import _obtain_keypoints
import videogpt_plus.model.wham_dataloader.constants as _C
from tqdm import tqdm
import torch
import argparse
def build_body_model(device, batch_size=1, gender='neutral', **kwargs):
    sys.stdout = open(os.devnull, 'w')
    body_model = SMPL(
        model_path=_C.BMODEL.FLDR,
        gender=gender,
        batch_size=batch_size,
        create_transl=False).to(device)
    sys.stdout = sys.__stdout__
    return body_model

def cache(args):
    device = f'cuda'
    print(device)

    num = args.device +args.now
    start = 1000 *num
    end = 3 * (num+1)
    detector = DetectionModel(device)
    extractor = FeatureExtractor(device, True)
    smpl = build_body_model(device)
    keypoints_normalizer = Normalizer()
    final_result = f'dataset/sft_data/cache_keypoints/cache_keypoints_{str(num)}_temp.pth' #a big dict: key: video_path, value: dict, keys: keypoints,init_kp,init_smpl, key_mask
    all_anno_file = 'dataset/wild_motion_videos/mergev1_caption.json'
    final_dict = {}
    with open(all_anno_file,'r') as f:
        all_anno = json.load(f)[start:end]
    for item in tqdm(all_anno):
        temp_dict = {}
        video_path = list(item.keys())[0]
        with torch.no_grad():
            keypoints,init_kp,init_smpl, key_mask = _obtain_keypoints(video_path,frame_resolution=224, max_frames=NUM_FRAMES,
                                                            num_video_frames=NUM_FRAMES,
                                                            num_context_images=NUM_CONTEXT_IMAGES,
                                                            detector=detector,extractor=extractor,
                                                            smpl=smpl,keypoints_normalizer=keypoints_normalizer)
            temp_dict['keypoints'] = keypoints.cpu()
            temp_dict['init_kp'] = init_kp.cpu()
            temp_dict['init_smpl'] = init_smpl.cpu()
            temp_dict['key_mask'] = key_mask.cpu()
            final_dict[video_path] = temp_dict
        del keypoints, init_kp, init_smpl, key_mask
        torch.cuda.empty_cache()
    # with open(final_result,'w') as f:
    #     json.dump(final_dict,f)
    torch.save(final_dict, final_result)
    # print(torch.load(final_result))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example script using argparse')


    # Optional argument with a default value
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--now', type=int, default=0)


    # Parse the arguments
    args = parser.parse_args()
    cache(args)
