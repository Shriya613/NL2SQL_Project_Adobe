import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load_file
from transformers import AutoTokenizer, AutoModel

# Custom class must match training definition
class BERTRewardModel(torch.nn.Module):
    def __init__(self, model_name="answerdotai/ModernBERT-base"):
        super().__init__()
        # Load base model
        self.bert = AutoModel.from_pretrained(
            model_name,
            reference_compile=False,
            attn_implementation="eager",
        )
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        pooled_output = torch.sum(last_hidden_state * attention_mask_expanded, dim=1) / \
                        torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)

        logits = self.classifier(pooled_output)
        scores = self.sigmoid(logits).squeeze(-1)

        return {"scores": scores, "logits": logits}


def _load_reward_state_dict(model_name: str):
    """Load the fine-tuned reward model weights from Hugging Face."""
    errors = []

    # Prefer safetensors when available, fall back to PyTorch binary weights
    candidates = (
        ("model.safetensors", lambda path: safe_load_file(path)),
        ("pytorch_model.bin", lambda path: torch.load(path, map_location="cpu")),
    )

    for filename, loader in candidates:
        try:
            # Refetch the model weights
            path = hf_hub_download(repo_id=model_name, filename=filename, force_download=True, resume_download=False)
            return loader(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{filename}: {exc}")

    raise RuntimeError(
        "Unable to load reward model weights from Hugging Face.\n" + "\n".join(errors)
    )


def _select_best_device():
    """Pick the CUDA device with the most free memory, fallback to CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(idx)
        except RuntimeError:
            free_mem, total_mem = 0, 0
        if free_mem > best_free:
            best_idx = idx
            best_free = free_mem
            best_total = total_mem

    gb = 1024 ** 3
    print(
        f"📈 Selected cuda:{best_idx} "
        f"({best_free / gb:.2f} / {best_total / gb:.2f} GiB free)"
    )
    return torch.device(f"cuda:{best_idx}")


def load_model_from_huggingface(model_name="DarianNLP/modernbert-nl-sql"):
    print(f"🔁 Loading tokenizer and model from HF: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Recreate architecture
    model = BERTRewardModel(model_name="answerdotai/ModernBERT-base")

    # Load HuggingFace weights (correct way)
    state_dict = _load_reward_state_dict(model_name)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "⚠ Warning while loading state dict",
            {"missing_keys": missing, "unexpected_keys": unexpected},
        )

    # Send to device
    device = _select_best_device()
    model.to(device).eval()

    print(f"✔ Loaded on {device}")
    return model, tokenizer, device
    
def get_reward(model, tokenizer, sql, reasoning, nl, device="cuda"):
    input_text = f"SQL: {sql}\nReasoning: {reasoning}\nNL: {nl}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,  # matches training
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return float(outputs["scores"].item())