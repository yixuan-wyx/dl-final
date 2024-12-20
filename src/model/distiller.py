"""
Distiller module
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM
from typing import Optional, List
from torch.nn import CrossEntropyLoss


class Distiller(nn.Module):
    """
    Distillation model with teacher-student combo

    Args Teacher: 
    - base_model (str): Big model after PEFT / Additional Reasoning.
    - draft_model (str): Draft model without fine-tuning.
    """
    def __init__(self, base_model:str, draft_model_name:str, args):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

        self.config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # load the pre-trained PEFT pairs
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights,
            torch_dtype=torch.float16,
            device_map={"":0}
        )

        print("Pretrained LoRA weights loaded for the teacher")

        # define the draft model
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        draft_model.config.use_cache = True
        draft_model = get_peft_model(draft_model, self.config)

        draft_model = draft_model.to(self.device)
        draft_model.print_trainable_parameters()

        # define the teacher and student
        self.main_dec = model
        self.draft_dec = draft_model
        self.freeze_teacher()


        # vocab_size
        self.vocab_size = self.main_dec.lm_head.out_features

        # loss attributes
        self.register_buffer("student_temp", torch.tensor(args.student_temp))
        self.register_buffer("teacher_temp", torch.tensor(args.teacher_temp))
        self.register_buffer("center_momentum", torch.tensor(args.center_momentum))
        self.register_buffer("lamda", torch.tensor(args.lamda))
        
        self.center = torch.zeros(1, 1, self.vocab_size).to(self.device)

    def freeze_teacher(self):
        for p in self.main_dec.parameters():
            p.requires_grad_(False)

    def distill_loss(self, main_logits:torch.Tensor, draft_logits:torch.Tensor):
        """
        """
        draft_logits = draft_logits / self.student_temp

        teacher_out = F.softmax((main_logits - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()

        loss = torch.sum(-teacher_out * F.log_softmax(draft_logits, dim=-1), dim=-1)
        return loss.mean()

    def js_loss(self, main_logits:torch.Tensor, draft_logits:torch.Tensor):
        main_probs = F.softmax(main_logits / self.student_temp, dim=-1)
        draft_probs = F.softmax(draft_logits / self.teacher_temp, dim=-1)

        # compute the mean probability
        mean_probs = (main_probs + draft_probs).mul(0.5)

        main_kl = main_probs * (main_probs - mean_probs)
        draft_kl = draft_probs * (draft_probs - mean_probs)

        js_loss = (main_kl + draft_kl).div(2.0)
        return js_loss.sum(dim=-1).mean()
    
    def reversekd(self, main_logits:torch.Tensor, draft_logits:torch.Tensor, labels):
        teacher_probs = F.softmax(main_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(draft_logits)

        logprobs = F.log_softmax(draft_logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)

        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (labels != -100).int()
        distill_loss = -torch.sum(x, dim=0) / torch.sum(mask.view(-1), dim=0)
        return distill_loss

    def compute_loss(self, main_logits:torch.Tensor, draft_logits:torch.Tensor, labels:torch.Tensor):
        # supervised loss
        shift_draft_logits = draft_logits[..., :-1, :].contiguous()

        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(main_logits.device)

        draft_loss_fct = CrossEntropyLoss()
        shift_draft_logits = shift_draft_logits.view(-1, self.vocab_size)

        sup_draft_loss = draft_loss_fct(shift_draft_logits, shift_labels)

        # dino-based distillation
        # distill_loss = self.distill_loss(main_logits, draft_logits)
        distill_loss = self.reversekd(main_logits, draft_logits, shift_labels)

        # total loss
        total_loss = sup_draft_loss + distill_loss.mul(self.lamda)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=[0,1], keepdim=True)
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    
    def forward(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None, 
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels = None
        ):
        
        base_out = self.main_dec(input_ids, attention_mask, position_ids)
        draft_out = self.draft_dec(input_ids, attention_mask, position_ids)

        base_logits = base_out["logits"]
        draft_logits = draft_out["logits"]

        loss = self.compute_loss(base_logits, draft_logits, labels)
        
        # update the momentum
        self.update_center(base_logits)

        return {
            "base_logits": base_logits,
            "draft_logits": draft_logits,
            "loss": loss
        }