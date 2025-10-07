import os

DATASET_DIR = os.environ.get("DATASET_DIR", "playground/data")

CC3M_595K = {
    "annotation_path": f"{DATASET_DIR}/pretraining/CC3M-595K/chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/CC3M-595K",
}

COCO_CAP = {
    "annotation_path": f"{DATASET_DIR}/pretraining/COCO/coco_cap_chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/COCO/train2014",
}

COCO_REG = {
    "annotation_path": f"{DATASET_DIR}/pretraining/COCO/coco_reg_chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/COCO/train2014",
}

COCO_REC = {
    "annotation_path": f"{DATASET_DIR}/pretraining/COCO/coco_rec_chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/COCO/train2014",
}

CONV_VideoChatGPT = {
    "annotation_path": f"{DATASET_DIR}/annotations/conversation_videochatgpt.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

VCG_HUMAN = {
    "annotation_path": f"{DATASET_DIR}/annotations/vcg_human_annotated.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

VCG_PLUS_112K = {
    "annotation_path": f"{DATASET_DIR}/annotations/vcg-plus_112K.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

CAPTION_VIDEOCHAT = {
    "annotation_path": f"{DATASET_DIR}/annotations/caption_videochat.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}

CLASSIFICATION_K710 = {
    "annotation_path": f"{DATASET_DIR}/annotations/classification_k710.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/k710",
}

CLASSIFICATION_SSV2 = {
    "annotation_path": f"{DATASET_DIR}/annotations/classification_ssv2.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/ssv2",
}

CONV_VideoChat1 = {
    "annotation_path": f"{DATASET_DIR}/annotations/conversation_videochat1.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/videochat_it",
}

REASONING_NExTQA = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_next_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/NExTQA",
}

REASONING_CLEVRER_QA = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_clevrer_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}

REASONING_CLEVRER_MC = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_clevrer_mc.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}

VQA_WEBVID_QA = {
    "annotation_path": f"{DATASET_DIR}/annotations/vqa_webvid_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}
HUMANML3D = {
    "annotation_path": "../resources/dataset/train_top1_t2m_new.json",
    "data_path": "",
}
# HUMANML3D = {
#     "annotation_path": "/mnt/data/nas/haidong/dataset/re_second/retrieval_from_motion_train.json",
#     "data_path": "",
# }

# HUMANML3D = {
#     "annotation_path": "/mnt/data/nas/haidong/dataset/sft_data/train_top1_kit_new_0.75.json",
#     "data_path": "",
# }
# HUMANML3D = {
#     "annotation_path": "/mnt/data/nas/haidong/dataset/sft_data/train_random_t2m.json",
#     "data_path": "",
# }
MERGEV1 = {
    "annotation_path": "/mnt/data/nas/haidong/dataset/pretrain/mergev1_pretrain_motionbert.json",
    "data_path": "",
}

# DPODEMO = {
#     "annotation_path": "/mnt/data/nas/haidong/dataset/dpo_selection/kit_r128_a256_bsz8x8_epoch16_new_train_3seed_self_fid0-5_458.json",
#     "data_path": "",
# }
# DPODEMO = {
#     "annotation_path": "/mnt/data/nas/haidong/dataset/dpo_selection/no_motion_nolora_bsz16x8_epoch6_decay0.02_train_3seed_self.json",
#     "data_path": "",
# }
# DPODEMO = {
#     "annotation_path": "/mnt/data/nas/haidong/dataset/dpo_selection/no_motion_r128_a256_bsz8x8_epoch3_random_train_3seed_self.json",
#     "data_path": "",
# }
DPODEMO = {
    "annotation_path": "../resources/dataset/no_motion_r128_a256_bsz8x8_epoch2_new_train_3seed_self.json",
    "data_path": "",
}
# DPODEMO = {
#     "annotation_path": "/mnt/data/nas/haidong/dataset/dpo_selection/kit_r128_a256_bsz8x8_epoch40_new_train_3seed_self.json",
#     "data_path": "",
# }
# DPODEMO = {
#     "annotation_path": "/mnt/data/nas/haidong/dataset/dpo_selection/kit_r128_a256_bsz8x8_epoch30_0.75_train_3seed_self.json",
#     "data_path": "",
# }
# DPODEMO = {
#     "annotation_path": "/mnt/data/nas/haidong/dataset/dpo_selection/kit_r128_a256_bsz8x8_epoch30_0.75_train_3seed_fid0.1_self.json",
#     "data_path": "",
# }