"""
LoRA for commonsense reasoning
"""

import os
import argparse
import sys
sys.path.append("../dl-project/")
import torch
import transformers

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel, LlamaForCausalLM, LlamaConfig
from src.utils.data import generate_and_tokenize_prompt
from src.utils.utils import str2bool
from peft import get_peft_model_state_dict, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from functools import partial

parser = argparse.ArgumentParser(description='LoRA for commensence reasoning')
parser.add_argument('--base_model', type=str, help='model architecture')
parser.add_argument('--data_path', type=str, help='data path')
parser.add_argument('--train_path', type=str, help='data path')
parser.add_argument('--valid_path', type=str, help='data path')

parser.add_argument('--output_dir', type=str, help='output directory')
parser.add_argument('--val_set_size', type=int, default=2000, help='size of the validation set')
parser.add_argument('--eval_step', type=int, default=200, help='evaluation step')
parser.add_argument('--save_step', type=int, default=200, help='save step')
parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff len')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning Rate')
parser.add_argument('--wd', type=float, default=0.0, help='Weight Decay')
parser.add_argument("--use_gradient_checkpointing", type=str2bool, nargs='?', const=True, default=False, help="enable gradient checkpoint")
parser.add_argument('--adapter', type=str, help='type of adaptor')

parser.add_argument('--lora_r', type=int, default=8, help='rank of lora')
parser.add_argument('--lora_alpha', type=int, default=16, help='alpha of lora')
parser.add_argument('--lora_dropout', type=float, default=0.05, help='dropout probability')
parser.add_argument('--target_modules', nargs='+', help='<Required> Set flag', required=True)


parser.add_argument("--use_wandb", type=str2bool, nargs='?', const=True, default=False, help="enable wandb")
parser.add_argument('--wandb_run_name', type=str, help='wandb run name')

parser.add_argument("--slice_sub", type=str2bool, nargs='?', const=True, default=False, help="enable wandb")
parser.add_argument('--slice_delta', type=int, default=5, help='number of layers to be trimmed')


args = parser.parse_args()

def main():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"world_size = {world_size}\n")
    ddp = world_size != 1

    # define the model
    model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
    tokenizer.padding_side = "left"  # Allow batched inference
    tokenizer.truncation=True
    tokenizer.padding=True

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_gradient_checkpointing)

    if args.adapter == "lora":
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    # get peft model
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # load the commonsence dataset
    if args.data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=args.data_path)
    else:
        data = load_dataset(args.data_path)

    # run through the test set
    if args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )

        train_data = (
            train_val["train"].shuffle().map(partial(generate_and_tokenize_prompt, tokenizer, True, args))
        )

        val_data = (
            train_val["test"].shuffle().map(partial(generate_and_tokenize_prompt, tokenizer, True, args))
        )
    else:
        train_data = data["train"].shuffle().map(partial(generate_and_tokenize_prompt, tokenizer, True, args))
        val_data = None

    # gradient accumulation
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.lr,
            weight_decay=args.wd,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=args.eval_step if args.val_set_size > 0 else None,
            save_steps=args.save_step,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if args.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=False,
            report_to="wandb" if args.use_wandb else None,
            run_name=args.wandb_run_name if args.use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True,
        ),
    )
    model.config.use_cache = False

    # start training
    print("Start Training!")
    trainer.train(resume_from_checkpoint=None)
    model.save_pretrained(args.output_dir, safe_serialization=False)
    

if __name__ == "__main__":
    main()