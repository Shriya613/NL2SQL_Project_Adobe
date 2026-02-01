import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Get the script directory and resolve paths relative to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
data_dir = os.path.join(project_root, "data")

# Load the dataset
df = pd.read_csv(os.path.join(data_dir, 'cot_dataset_with_corruptions.csv'))

# Separate corrupted and non-corrupted examples
corrupted = df[df['is_corrupted'] == True]['similarity_with_penalty']
non_corrupted = df[df['is_corrupted'] == False]['similarity_with_penalty']

# Create the plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram
ax1 = axes[0]
ax1.hist(non_corrupted, bins=50, alpha=0.7, label=f'Non-corrupted (n={len(non_corrupted)})', 
         color='green', edgecolor='black')
ax1.hist(corrupted, bins=50, alpha=0.7, label=f'Corrupted (n={len(corrupted)})', 
         color='red', edgecolor='black')
ax1.set_xlabel('Similarity Score')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Similarity Scores')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plot
ax2 = axes[1]
ax2.boxplot([non_corrupted, corrupted], tick_labels=['Non-corrupted', 'Corrupted'])
ax2.set_ylabel('Similarity Score')
ax2.set_title('Similarity Score Comparison')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(script_dir, 'similarity_distribution.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved plot to {output_path}")

# Print statistics
print("\n📊 Statistics:")
print(f"\nNon-corrupted examples:")
print(f"  Count: {len(non_corrupted)}")
print(f"  Mean: {non_corrupted.mean():.4f}")
print(f"  Std: {non_corrupted.std():.4f}")
print(f"  Median: {non_corrupted.median():.4f}")
print(f"  Min: {non_corrupted.min():.4f}")
print(f"  Max: {non_corrupted.max():.4f}")

print(f"\nCorrupted examples:")
print(f"  Count: {len(corrupted)}")
print(f"  Mean: {corrupted.mean():.4f}")
print(f"  Std: {corrupted.std():.4f}")
print(f"  Median: {corrupted.median():.4f}")
print(f"  Min: {corrupted.min():.4f}")
print(f"  Max: {corrupted.max():.4f}")

print(f"\nDifference in means: {non_corrupted.mean() - corrupted.mean():.4f}")
print(f"\nCorruption rate: {len(corrupted)/(len(corrupted)+len(non_corrupted))*100:.2f}%")

# Show highest scoring corruptions
print("\n" + "="*80)
print("HIGHEST SCORING CORRUPTIONS (most deceptive):")
print("="*80)
corrupted_df = df[df['is_corrupted'] == True].copy()
top_corrupted = corrupted_df.nlargest(10, 'similarity_with_penalty')

for idx, row in top_corrupted.iterrows():
    print(f"\n--- Similarity: {row['similarity_with_penalty']:.4f} (Corruptions: {row['corruption_count']}) ---")
    print(f"SQL: {row['sql'][:150]}...")
    print(f"Predicted NL: {row['predicted_nl']}")
    print(f"True NL: {row['true_nl']}")

