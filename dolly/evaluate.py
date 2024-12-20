"""
Execution entrance
"""

import yaml
import torch
import torch.nn.functional as F
import argparse
import datasets

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer

parser = argparse.ArgumentParser(description='MiniLLM')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()


def get_model(teacher:str, model:str):
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    student_model = AutoModelForCausalLM.from_pretrained(
        model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(teacher, trust_remote_code=True)

    return teacher_model, student_model, tokenizer


def get_distil_loss(tokenizer, model, teacher_model, model_batch, no_model_batch, logits):
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(**model_batch, use_cache=False)
        teacher_logits = teacher_outputs.logits

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    
    return distil_loss

def prepare_hf_dataset(dataset_name, split="train"):
    print(f"Loading data {dataset_name}...")
    dataset = datasets.load_dataset(dataset_name, split=split)
    return dataset

def prepare_dataloader(dataset, batch_size:int):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    return dataloader


def main():
    with open(args.config_dir, 'r') as f:
        config = yaml.full_load(f)

    teacher_name = config["teacher_name"]
    student_name = config["student_name"]

    teacher_model, student_model, tokenizer = get_model(teacher_name, student_name)
    
    trainset = prepare_hf_dataset("MiniLLM/dolly-processed", split="train")
    validset = prepare_hf_dataset("MiniLLM/dolly-processed", split="validation")

    
    # generation config
    do_sample = config["eval"]["do_sample"]
    top_p = config["eval"]["top_p"]
    top_k = config["eval"]["top_k"]
    temperature = config["eval"]["temperature"]
    no_repeat_ngram_size = config["eval"]["no_repeat_ngram_size"]
    max_length = config["eval"]["max_length"]

    generation_config = GenerationConfig(
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=None,
        max_length=max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )

    rl_score = []
    
    pbar = tqdm(validset)
    for idx, batch in enumerate(pbar):
        text = batch["instruction"]
        reference = batch["output"]

        input_text = tokenizer(
            text,
            padding=False,
            add_special_tokens = True,
            return_tensors = "pt"
        )

        input_ids = input_text["input_ids"].to(student_model.device)

        output = student_model.generate(
            input_ids=input_ids,
            generation_config=generation_config
        )
        output_seq = output[0][0]
        response = output_seq[input_ids.shape[1]:]

        # Decode the generated text
        generated_text = tokenizer.decode(response, skip_special_tokens=True)

        # Evaluate ROUGE score
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, generated_text)

        rl = scores["rougeL"].precision
        rl_score.append(rl)
        pbar.set_description(f"RL Score = {rl:.3f}")

    print(f"Average R-L Score = {sum(rl_score) / len(rl_score)}")

if __name__ == "__main__":
    main()
