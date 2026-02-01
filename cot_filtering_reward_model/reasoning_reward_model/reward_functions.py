import re
import torch
import torch.nn.functional as F
import numpy as np

import re


def reward_func(completions, sbert_similarity: list[float], **kwargs) -> list[float]:
    """
    Combined format + content reward (TRL / GPRO compatible).
    Returns a list of floats in [-1, 1].

    Logic:
      - If output format invalid or score not in [0, 1] → reward = -1.0
      - Otherwise reward = 1 - 2 * (predicted_score - true_score)^2
        (perfect match → +1, opposite ends → -1)
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    # More flexible pattern matching
    patterns = [
        r"^<think>\n.*?\n</think>\n<score>\n(.*?)\n</score>\n$",  # Original format
        r"<think>.*?</think>\s*<score>\s*(.*?)\s*</score>",      # More flexible spacing
        r"<score>\s*(.*?)\s*</score>",                           # Just score tag
    ]

    for i, r in enumerate(responses):
        score_val = None
        
        # Check if reasoning tag exists first
        has_reasoning = bool(re.search(r'<(think|redacted_reasoning)>', r, re.IGNORECASE))
        
        if not has_reasoning:
            rewards.append(-1.0)
            continue
        
        # Try different patterns
        for pattern in patterns:
            match = re.search(pattern, r.strip(), re.DOTALL)
            if match:
                score_str = match.group(1).strip()
                try:
                    score_val = float(score_str)
                    break
                except ValueError:
                    continue
        
        # If no valid score found, give penalty
        if score_val is None:
            rewards.append(-1.0)
            continue

        # Validate score range
        if not (0.0 <= score_val <= 1.0):
            rewards.append(-1.0)
            continue
        err = abs(score_val - sbert_similarity[i])
        reward = 2.0 * np.exp(-2 * err) - 1.0
        # Exponential reward decay: sharply penalizes even small errors, stable and smooth
        # Error → Reward mapping (approximate):
        #  Error   Reward
        #  0.0     +1.00
        #  0.05    +0.8
        #  0.1     +0.62
        #  0.15    +0.46
        #  0.2     +0.32
        #  0.3     +0.07
        #  0.4     -0.12
        #  0.7     -0.53
        #  1.0     -0.75
        rewards.append(float(torch.clamp(torch.tensor(reward), -1.0, 1.0)))

    return rewards


def debug_reward_func(completions, sbert_similarity: list[float], **kwargs) -> list[float]:
    """Debug version that prints information about rewards"""
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    print(f"\n=== REWARD DEBUG ===")
    print(f"Number of completions: {len(completions)}")
    print(f"Number of similarities: {len(sbert_similarity)}")
    
    for i, (r, sim) in enumerate(zip(responses, sbert_similarity)):
        print(f"\nCompletion {i+1}:")
        print(f"Response: {r[:200]}...")
        print(f"True similarity: {sim}")
        
        # Try to extract score
        patterns = [
            r"<score>\s*(.*?)\s*</score>",
            r"<think>.*?</think>\s*<score>\s*(.*?)\s*</score>",
        ]
        
        score_val = None
        for pattern in patterns:
            match = re.search(pattern, r.strip(), re.DOTALL)
            if match:
                score_str = match.group(1).strip()
                try:
                    score_val = float(score_str)
                    break
                except ValueError:
                    continue
        
        if score_val is None:
            print(f"❌ No valid score found")
            rewards.append(-1.0)
        elif not (0.0 <= score_val <= 1.0):
            print(f"❌ Score out of range: {score_val}")
            rewards.append(-1.0)
        else:
            mse = (score_val - sim) ** 2
            reward = 1.0 - 2.0 * mse
            print(f"✅ Score: {score_val}, MSE: {mse:.4f}, Reward: {reward:.4f}")
            rewards.append(float(torch.clamp(torch.tensor(reward), -1.0, 1.0)))
    
    print(f"Final rewards: {rewards}")
    return rewards
    