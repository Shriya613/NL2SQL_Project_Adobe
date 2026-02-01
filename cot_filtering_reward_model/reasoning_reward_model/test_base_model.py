import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from reward_functions import reward_func

# Load dataset
df = pd.read_csv('../../data/cot_dataset_with_corruptions.csv')
print(f"Loaded {len(df)} examples from dataset")

# Load BASE model (untrained)
print("\nLoading BASE (untrained) model...")
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)
model.eval()
print("Model loaded successfully!")

# Test on 5 examples
num_tests = 5
print(f"\nGenerating predictions on {num_tests} examples with BASE (untrained) model...")
print("="*80)

for i in range(num_tests):
    row = df.iloc[i]
    
    prompt = row['prompt']
    real_score = row['similarity_with_penalty']
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the completion (remove the prompt)
    completion = generated_text[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
    
    # Calculate reward
    completions = [[{"content": completion}]]
    similarities = [real_score]
    rewards = reward_func(completions, similarities)
    reward = rewards[0]
    
    # Extract predicted score
    import re
    match = re.search(r'<score>\s*(.*?)\s*</score>', completion, re.DOTALL)
    if match:
        try:
            predicted_score = float(match.group(1).strip())
            mse = (predicted_score - real_score) ** 2
        except ValueError:
            predicted_score = None
            mse = None
    else:
        predicted_score = None
        mse = None
    
    # Extract reasoning
    reasoning_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = "No reasoning found"
    
    print()
    print("-"*80)
    print(f"EXAMPLE {i+1}")
    print("-"*80)
    print(f"Real Score: {real_score:.4f}")
    if predicted_score is not None:
        print(f"Predicted Score: {predicted_score:.4f}")
        print(f"MSE: {mse:.6f}")
    else:
        print(f"Predicted Score: None (invalid format)")
        print(f"MSE: N/A")
    print(f"Reward: {reward:.4f}")
    print(f"\nReasoning:")
    print(reasoning[:300] + "..." if len(reasoning) > 300 else reasoning)
    print()

print("="*80)
print("TEST COMPLETE")
print("="*80)

