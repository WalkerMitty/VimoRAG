export DATASET_DIR=../resources/playground/data
export PYTHONPATH="./:$PYTHONPATH"
BASE_LLM_PATH=../resources/playground/Phi-3-mini-4k-instruct
OUTPUT_DIR_PATH=../output/dpo_model_yours
SFT_MODEL='../output/sft_model/merged_lora'
deepspeed mcdpo/train/train_with_dpo.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$BASE_LLM_PATH" \
    --sft_model "$SFT_MODEL" \
    --output_dir $OUTPUT_DIR_PATH \
    --dataset_use DPO \
    --version phi3_instruct \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --dataloader_num_workers 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --bf16 True \
    --logging_steps 1 \
    --num_train_epochs 1 \
    --model_max_length 4096 \
    --lazy_preprocess True \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --tf32 True \
    --lr_scheduler_type "cosine" \
