
# DATA_PATH="JSON FILE PATH"
# IMAGE_FOLDER="IMAGE FOLDER PATH"
# run_name="RUN NAME"


DATA_PATH=/nfs/turbo/justincj-turbo/ancao/repos/ImprovingVLM/util_scripts/image_dpo_cambrian_8b_all_False_20_16_corrupted_score5_80k.json
IMAGE_FOLDER=/nfs/turbo/justincj-turbo/tiangel/improvingVLM/
run_name=llava-7b-image-dpo-all-80-blur-64-pixelate-corrupted-80k-lora

ouput_dir=./checkpoints/${run_name}
# Notice that I am loading the latest model checkopint 
model_name=liuhaotian/llava-v1.5-7b # Use the previous model checkpoint
deepspeed llava/train/train_mem_dpo_dpotrainer_image.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${model_name} \
    --version v1 \
    --dpo_alpha 1.0 --beta 0.5 --gamma 0 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${ouput_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 5e-7\
    --is_multimodal True \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --freeze_mm_mlp_adapter True 
