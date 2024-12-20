dataset="piqa"
logits_prob=0.01
label="OPT-6.7B+1.3B_ReverseDistill${logits_prob}"
peft="lora"
run_dir="save/distill_${peft}_${dataset}_${label}_${logits_prob}_rank8/"

model_name="facebook/opt-6.7b"
draft_model_name="facebook/opt-1.3b"

student_temp=0.1
teacher_temp=0.1
lora_weights="save/facebook/opt-6.7b_lora_piqa_rank8_qkv_up_and_down_proj/checkpoint-2000/"

clear;python3 ./pedistill/main.py \
    --base_model ${model_name} \
    --draft_model ${draft_model_name} \
    --data_path "commonsense_reasoning/dataset/${dataset}/train.json" \
    --batch_size 8 \
    --logging "training.log" \
    --run_dir ${run_dir} \
    --student_temp ${student_temp} \
    --student_temp ${teacher_temp} \
    --lora_weights ${lora_weights} \

