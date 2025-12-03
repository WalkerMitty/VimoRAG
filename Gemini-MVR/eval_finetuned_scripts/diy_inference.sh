# NUM=0
# PORT=12330
# DEVICE=$((NUM % 8))
# CUDA_VISIBLE_DEVICES=$DEVICE torchrun --nproc_per_node=1 --master_port $((PORT + NUM)) -m retrieval_from_wild \
python -u -m retrieval_from_wild \
    --do_eval \
    --num_thread_reader=4 \
    --n_display=50 \
    --val_csv="../output/demo_input.json" \
    --lr=1e-4 \
    --max_words=77 \
    --max_frames=16 \
    --batch_size_val=8 \
    --datatype="motionx" \
    --feature_framerate=1 \
    --slice_framepos=2 \
    --linear_patch=2d \
    --pretrained_clip_name="ViT-L/14" \
    --clip_evl \
    --pretrained_path="../resources/InternVideo-MM-L-14.ckpt" \
    --finetuned_path="../resources/eventmodel_bsz128_lr1e-3_epoch5_v1_data0.1_new/pytorch_model.bin" \
    --finetuned_path_object="../resources/object_epoch5_bsz128_coeflr4e-3/pytorch_model.bin" \
    --finetuned_path_action="../resources/motionbert_bsz2048_lr1e-4_epoch10_scale100_two_loss/pytorch_model.bin" \
    --output_dir="./diy_output/temp" \
    --mergeclip=True \
    --inference_result="./diy_output/retrieval_result.json" \
    --k=1 \
    --saved_video_embed="../resources/retrieval_inference_wild/middle_t2m_new.pth" \
    --query_text_file="../output/demo_input.json" \
    --train_tower="event" \
    --verb_model="internvideo" \
    --action_model="motionbert"