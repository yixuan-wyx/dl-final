"""

"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from typing import Optional, List
from torch.nn import CrossEntropyLoss


class Wrap(nn.Module):
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
        # get peft model
        model = get_peft_model(model, self.config)
        model.print_trainable_parameters()

        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        draft_model.config.use_cache = True
        draft_model = get_peft_model(draft_model, self.config)

        draft_model = draft_model.to(self.device)
        draft_model.print_trainable_parameters()

        # isolate the decoder
        self.main_dec = model
        self.draft_dec = draft_model
        # assert self.main_dec.vocab_size == self.draft_dec.vocab_size, "Vocab size of two models must be the same!"

        # vocab_size
        self.vocab_size = self.main_dec.lm_head.out_features
        self.topk = int(self.vocab_size * args.logits_prob)

        # loss
        self.corr_reg = args.corr_reg
        self.draft_loss = args.draft_loss
        self.use_vocab_loss = args.use_vocab_loss
        self.use_dual_kl_loss = args.use_dual_kl_loss
        self.use_kl_loss = args.use_kl_loss
        
        # loss attributes
        self.register_buffer("student_temp", torch.tensor(args.student_temp))
        self.register_buffer("teacher_temp", torch.tensor(args.teacher_temp))
        self.register_buffer("alpha", torch.tensor(args.alpha))

    def normalize_feature(self, tensor:torch.Tensor):
        mean = tensor.mean(dim=-1, keepdim=True)
        var = tensor.var(dim=-1, keepdim=True)
        tensor = (tensor - mean) / (var + 1e-6)**0.5
        return tensor

    def corr_loss(self, main_logits:torch.Tensor, draft_logits:torch.Tensor):
        N, D, K = main_logits.shape

        # normalize features
        main_logits = self.normalize_feature(main_logits)
        draft_logits = self.normalize_feature(draft_logits)

        corr = torch.einsum("bik, bjk -> bij", main_logits, draft_logits).div(K).mean(dim=0)
        diag = torch.eye(D, device=corr.device)
        corr_loss = 1 - corr[diag.bool()]

        return corr_loss.sum().mul(self.corr_reg)
    
    def vocab_loss(self, main_logits:torch.Tensor, draft_logits:torch.Tensor):
        N, D, K = main_logits.shape

        # compute probability
        main_prob = torch.nn.functional.softmax(main_logits, dim=-1)
        draft_prob = torch.nn.functional.softmax(draft_logits, dim=-1)

        # slice top-K
        sliced_main_logits, _ = torch.topk(main_prob, self.topk)
        sliced_draft_logits, _ = torch.topk(draft_prob, self.topk)

        corr = torch.einsum("bki, bkj -> bij", sliced_main_logits, sliced_draft_logits).div(D)
        corr_loss = 1.0 - corr
        return corr_loss.mean().mul(self.corr_reg)

    def kl_loss(self, main_logits:torch.Tensor, draft_logits:torch.Tensor):

        draft_logits = draft_logits / self.student_temp
        main_logits = main_logits / self.teacher_temp

        student_out = F.softmax(draft_logits, dim=-1)
        teacher_out = F.softmax(main_logits, dim=-1)

        student_out = F.log_softmax(student_out, dim=-1)
        teacher_out = F.log_softmax(teacher_out, dim=-1)

        kl_loss = teacher_out * (teacher_out - student_out)
        return kl_loss.sum(dim=-1).mean()
    
    def invar_loss(self, draft_logits:torch.Tensor):
        """
        Minimize the probability aliasing
        """
        N, D = draft_logits.size(0), draft_logits.size(2) 
        
        student_out = F.softmax(draft_logits, dim=-1)
        top_k_logits, _ = torch.topk(student_out, self.topk)

        std_x = torch.sqrt(top_k_logits.var(dim=-1) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x))

        return std_loss.mul(self.corr_reg)
    
    def js_loss(self, main_logits:torch.Tensor, draft_logits:torch.Tensor):
        main_probs = F.softmax(main_logits / self.student_temp, dim=-1)
        draft_probs = F.softmax(draft_logits / self.teacher_temp, dim=-1)

        # compute the mean probability
        mean_probs = (main_probs + draft_probs).mul(0.5)

        main_kl = main_probs * (main_probs - mean_probs)
        draft_kl = draft_probs * (draft_probs - mean_probs)

        js_loss = (main_kl + draft_kl).div(2.0)
        return js_loss.sum(dim=-1).mean()
    
    def compute_loss(self, main_logits:torch.Tensor, draft_logits:torch.Tensor, labels:torch.Tensor):
        
        # supervised loss
        shift_main_logits = main_logits[..., :-1, :].contiguous()
        shift_draft_logits = draft_logits[..., :-1, :].contiguous()

        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(main_logits.device)

        main_loss_fct = CrossEntropyLoss()
        draft_loss_fct = CrossEntropyLoss()

        shift_main_logits = shift_main_logits.view(-1, self.vocab_size)
        shift_draft_logits = shift_draft_logits.view(-1, self.vocab_size)
        
        sup_main_loss = main_loss_fct(shift_main_logits, shift_labels)
        sup_draft_loss = draft_loss_fct(shift_draft_logits, shift_labels)

        sup_loss = sup_main_loss + sup_draft_loss

        if not self.use_vocab_loss:
            corr_loss = self.corr_loss(main_logits, draft_logits)
        else:
            if self.use_kl_loss:
                corr_loss = self.kl_loss(main_logits, draft_logits)
            
            elif self.use_dual_kl_loss:
                main_sfl = main_logits.detach()
                draft_sfl = draft_logits.detach()

                main_corr_loss = self.kl_loss(draft_sfl, main_logits)
                draft_corr_loss = self.kl_loss(main_sfl, draft_logits)

                invar_loss = self.invar_loss(draft_logits)

                corr_loss = (main_corr_loss + draft_corr_loss).div(2) + invar_loss
            elif self.use_js_loss:
                corr_loss = self.js_loss(main_logits, draft_logits)
            else:
                corr_loss = self.vocab_loss(main_logits, draft_logits)

        total_loss = sup_loss.mul(1 - self.alpha) + corr_loss.mul(self.alpha)

        return total_loss

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
        
        base_out = self.main_dec(
            input_ids,
            attention_mask,
            position_ids,
        )

        draft_out = self.draft_dec(
            input_ids,
            attention_mask,
            position_ids,
        )

        base_logits = base_out["logits"]
        draft_logits = draft_out["logits"]

        loss = self.compute_loss(base_logits, draft_logits, labels)

        return {
            "base_logits": base_logits,
            "draft_logits": draft_logits,
            "loss": loss
        }
