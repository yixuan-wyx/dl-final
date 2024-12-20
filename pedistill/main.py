"""
"""
import os
import sys
sys.path.append("../dl-project/")

import torch
import argparse
import logging
import transformers

from functools import partial
from datasets import load_dataset, Dataset
from src.model.wrap import Wrap
from src.model.distiller import Distiller
from src.utils.data import generate_and_tokenize_prompt

from transformers import AutoTokenizer
from src.trainer.base import Trainer

parser = argparse.ArgumentParser(description='LoRA for commensence reasoning')

parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training ')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loading')
parser.add_argument('--base_model', type=str, help='model architecture')
parser.add_argument('--draft_model', type=str, help='draft model architecture')
parser.add_argument('--data_path', type=str, help='data path')
parser.add_argument('--val_set_size', type=int, default=2000, help='size of the validation set')
parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff len')
parser.add_argument('--run_dir', type=str, default="./save/", help='run directory')
parser.add_argument('--logging', type=str, default="training.log", help='Logging path')

# Wrap Args
parser.add_argument('--draft_loss', action='store_true')
parser.add_argument('--use_vocab_loss', action='store_true')
parser.add_argument('--use_kl_loss', action='store_true')
parser.add_argument('--use_js_loss', action='store_true')
parser.add_argument('--use_dual_kl_loss', action='store_true')
parser.add_argument('--logits_prob', type=float, default=0.05, help='Probability cutoff for logits correlation')

# Distiller Args
parser.add_argument('--student_temp', type=float, default=0.1)
parser.add_argument('--teacher_temp', type=float, default=0.1)
parser.add_argument('--lamda', type=float, default=1e-3)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--corr_reg', type=float, default=1e-3)
parser.add_argument('--center_momentum', type=float, default=0.9)

# PEFT Args
parser.add_argument('--lora_r', type=int, default=8, help='rank of lora')
parser.add_argument('--lora_alpha', type=int, default=16, help='alpha of lora')
parser.add_argument('--lora_dropout', type=float, default=0.05, help='dropout probability')
parser.add_argument('--lora_weights', type=str, help='directory to lora weights')


args = parser.parse_args()

def main():
    # folder
    if not os.path.isdir(args.run_dir):
        os.makedirs(args.run_dir, exist_ok=True)

    # initialize terminal logger
    logger = logging.getLogger("training")
    fileHandler = logging.FileHandler(os.path.join(args.run_dir, args.logging))
    fileHandler.setLevel(0)
    logger.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)

    model = Distiller(args.base_model, args.draft_model, args=args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
    tokenizer.padding_side = "left"  # Allow batched inference

    print(model.main_dec)
    
    if args.data_path.endswith(".json"):
        dataset = load_dataset("json", data_files=args.data_path)
    else:
        dataset = load_dataset(args.data_path)

    # run through the test set
    if args.val_set_size > 0:
        train_val = dataset["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )

        train_data = (
            train_val["train"].shuffle().map(partial(generate_and_tokenize_prompt, tokenizer, True, args))
        )

        val_data = (
            train_val["test"].shuffle().map(partial(generate_and_tokenize_prompt, tokenizer, True, args))
        )
    else:
        train_data = dataset["train"].shuffle().map(partial(generate_and_tokenize_prompt, tokenizer, True, args))
        val_data = None

    flag = isinstance(train_data, Dataset)
    iter_flag = isinstance(train_data, torch.utils.data.IterableDataset)
    print(f"Type of the dataset: {type(train_data)} | is_dataset = {flag} | is_iter_data: {iter_flag}")

    trainer = Trainer(
        model = model,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collector=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        args=args,
        logger=logger
    )

    # test dataloader initialization
    trainer.train()

if __name__ == "__main__":
    main()