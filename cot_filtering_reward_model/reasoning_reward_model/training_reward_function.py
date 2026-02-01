import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from reward_functions import reward_func
from sklearn.model_selection import train_test_split
from datasets import Dataset


def validate_dataset(df):
    """Basic validation."""
    print("🔍 Validating dataset...")
    required_cols = ['sql','reasoning','predicted_nl','true_nl','similarity_with_penalty','prompt']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.dropna(subset=required_cols)
    df = df[df['prompt'].str.len() > 0]
    print(f"✅ Valid examples: {len(df)}")
    return df


def main():
    data_csv = "../../data/cot_dataset_with_corruptions.csv"
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"❌ Dataset not found at {data_csv}")

    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} examples")
    df = validate_dataset(df)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train examples: {len(df_train)}, Test examples: {len(df_test)}")

    # ---- GPU summary ----
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"🧠 Detected {n} GPUs")
        for i in range(n):
            free_gb = torch.cuda.mem_get_info(i)[0] / 1024**3
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} — {free_gb:.2f} GB free")
    else:
        print("⚠️  CUDA not available, using CPU")

    # ---- Model ----
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    output_dir = "outputs/gpro_reward_function_multi_full"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🧩 Loading model in bf16 across all GPUs (no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",            # automatically shard layers across GPUs
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    # ---- LoRA ----
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
        task_type="CAUSAL_LM",
        bias="none",
        lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"✅ Model distributed across devices: {set(p.device for p in model.parameters())}")

    # ---- GRPO config ----
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name="GRPO-Reward-FullPrecision",
        learning_rate=2e-5,
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=512,
        num_train_epochs=2,
        save_steps=10,
        max_grad_norm=0.1,
        remove_unused_columns=False,
        report_to="none",
    )

    # ---- Dataset ----
    dataset = Dataset.from_dict({
        'prompt': df_train['prompt'].tolist(),
        'similarity_with_penalty': df_train['similarity_with_penalty'].tolist(),
    })

    def gpro_reward_func(completions, **kwargs):
        sims = kwargs.get('similarity_with_penalty', [])
        formatted = [[{"content": c}] for c in completions]
        return reward_func(formatted, sims)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[gpro_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print("🚀 Starting GRPO training (multi-GPU, no quantization)...")
    trainer.train()

    # ---- Save ----
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
