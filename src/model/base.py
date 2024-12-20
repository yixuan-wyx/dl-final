"""
Base Model Wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Wrap(nn.Module):
    def __init__(self, big_model: nn.Module, small_model: nn.Module):
        super().__init__()
        self.big_model = big_model
        self.small_model = small_model

    def compute_loss(self, zb: torch.Tensor, zs: torch.Tensor):
        # Calculate the MSE loss between the outputs of the big model and small model
        mse_loss = F.mse_loss(zs, zb)
        return mse_loss
    
    def forward(self, **inputs):
        print(f"Input shapes: {[v.shape for v in inputs.values()]}")
        if 'input_ids' not in inputs:
            raise ValueError("Expected 'input_ids' in the inputs.")

        obig = self.big_model(**inputs)
        osmall = self.small_model(**inputs)

        loss = self.compute_loss(obig.logits, osmall.logits)
        
        return {"loss": loss, "logits": osmall.logits}
    

class DraftWrap(nn.Module):
    def __init__(self, model:nn.Module):
        self.model = model

    def forward(self, **inputs):
        output = self.model(**inputs)
        return output

