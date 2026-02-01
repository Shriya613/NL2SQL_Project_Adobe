"""
Utility module for loading and running the finetuned ModernBERT reward model.

The model mirrors the architecture defined in `mosaic_bert_training.py`:
 - base encoder: answerdotai/ModernBERT-base (8k context support)
 - pooling: attention-mask-weighted mean pooling
 - head: single linear layer + sigmoid to output a score in [0, 1]
"""

import os
from typing import Optional

import torch
from transformers import AutoModel


class BERTRewardModel(torch.nn.Module):
    """ModernBERT encoder with a sigmoid regression head."""

    def __init__(self, model_name: str = "answerdotai/ModernBERT-base"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
            reference_compile=False,
            attn_implementation="eager",
        )
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels: Optional[torch.Tensor] = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_hidden = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_hidden / sum_mask

        logits = self.classifier(pooled_output)
        scores = self.sigmoid(logits).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(scores, labels) * 100

        return {"loss": loss, "scores": scores, "logits": logits}


def load_finetuned_model(model_dir: str, device: Optional[str] = None) -> BERTRewardModel:
    """
    Load the finetuned ModernBERT reward model from `model_dir`.

    Args:
        model_dir: Path containing model.safetensors (preferred) or pytorch_model.bin.
        device: Optional torch device string. Defaults to CUDA if available else CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    model = BERTRewardModel()
    model.to(torch_device)

    state_dict = None
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path)
        except ImportError as exc:
            print(f"⚠️  safetensors not available ({exc}); falling back to pytorch_model.bin if present.")

    if state_dict is None and os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location=torch_device)

    if state_dict is None:
        raise FileNotFoundError(
            f"Could not find model weights in {model_dir}. "
            "Expected either model.safetensors or pytorch_model.bin."
        )

    model.load_state_dict(state_dict)
    model.eval()
    return model

