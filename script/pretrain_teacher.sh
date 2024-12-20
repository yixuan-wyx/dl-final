export CUDA_VISIBLE_DEVICES=0
dataset="piqa"
model_name="facebook/opt-6.7b"

torchrun --nproc_per_node=1 --master_port 48001 ./commonsense_reasoning/main.py \
    --base_model ${model_name} \
    --data_path "/share/seo/llm_datasets/commonsense/dataset/${dataset}/train.json" \
    --output_dir "save/${model_name}_lora_${dataset}_rank8_qkv_up_and_down_proj/" \
    --lora_r 8 \
    --lora_alpha 16 \
    --batch_size 4 \
    --micro_batch_size 4 \
    --lora_dropout 0.05 \
    --target_modules "q_proj" "k_proj" "v_proj" \
    --use_gradient_checkpointing True \
    --adapter "lora" \