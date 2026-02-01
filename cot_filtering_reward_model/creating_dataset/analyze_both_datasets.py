import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Get the script directory and resolve paths relative to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
data_dir = os.path.join(project_root, "data")

print("="*80)
print("ANALYZING: cot_dataset_with_corruptions.csv")
print("="*80)

# Load dataset 1
df1 = pd.read_csv(os.path.join(data_dir, 'cot_dataset_with_corruptions.csv'))
corrupted1 = df1[df1['is_corrupted'] == True]['similarity_with_penalty']
non_corrupted1 = df1[df1['is_corrupted'] == False]['similarity_with_penalty']

print(f"\nTotal examples: {len(df1)}")
print(f"Corrupted: {len(corrupted1)} ({len(corrupted1)/len(df1)*100:.2f}%)")
print(f"Non-corrupted: {len(non_corrupted1)} ({len(non_corrupted1)/len(df1)*100:.2f}%)")
print(f"\nCorrupted - Mean: {corrupted1.mean():.4f}, Median: {corrupted1.median():.4f}")
print(f"Non-corrupted - Mean: {non_corrupted1.mean():.4f}, Median: {non_corrupted1.median():.4f}")

print("\n" + "="*80)
print("ANALYZING: bird_sql2nl_cot_dataset.csv")
print("="*80)

# Load dataset 2
df2 = pd.read_csv(os.path.join(data_dir, 'bird_sql2nl_cot_dataset.csv'))
similarity_col = 'sbert_similarity' if 'sbert_similarity' in df2.columns else 'similarity_with_penalty'
similarities2 = df2[similarity_col]

print(f"\nTotal examples: {len(df2)}")
print(f"Mean similarity: {similarities2.mean():.4f}")
print(f"Median similarity: {similarities2.median():.4f}")
print(f"Std: {similarities2.std():.4f}")
print(f"Min: {similarities2.min():.4f}")
print(f"Max: {similarities2.max():.4f}")

# Create comparison plots (3 columns, 2 rows)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Dataset 1 histogram (separated)
ax1 = axes[0, 0]
# Plot histograms with new colors
n1, bins1, patches1 = ax1.hist(non_corrupted1, bins=50, alpha=0.85, label=f'Non-corrupted (n={len(non_corrupted1)})', 
         color='#e3eaff', edgecolor='black')
n2, bins2, patches2 = ax1.hist(corrupted1, bins=50, alpha=0.85, label=f'Corrupted (n={len(corrupted1)})', 
         color='#fdc5af', edgecolor='black')
# Add line tracing overall shape
all_scores1 = df1['similarity_with_penalty']
n_combined, bins_combined = np.histogram(all_scores1, bins=50)
bin_centers = (bins_combined[:-1] + bins_combined[1:]) / 2
ax1.plot(bin_centers, n_combined, color='#9ab0e0', linewidth=1.5, alpha=0.8, label='Overall shape')
ax1.set_xlabel('Similarity Score')
ax1.set_ylabel('Frequency')
ax1.set_title('Dataset 1: Separated Views')
ax1.legend(fontsize=18)
ax1.grid(True, alpha=0.3)

# Dataset 1 combined histogram
ax2 = axes[0, 1]
ax2.hist(all_scores1, bins=50, alpha=0.7, color='purple', edgecolor='black')
ax2.set_xlabel('Similarity Score')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Dataset 1: Combined (n={len(df1)})')
ax2.grid(True, alpha=0.3)

# Dataset 2 histogram
ax3 = axes[0, 2]
ax3.hist(similarities2, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax3.set_xlabel('Similarity Score')
ax3.set_ylabel('Frequency')
ax3.set_title('Dataset 2: bird_sql2nl_cot_dataset')
ax3.grid(True, alpha=0.3)

# Dataset 1 box plot (separated)
ax4 = axes[1, 0]
bp4 = ax4.boxplot([non_corrupted1, corrupted1], tick_labels=['Non-corrupted', 'Corrupted'])
ax4.set_ylabel('Similarity Score')
ax4.set_title('Dataset 1: Comparison')
ax4.grid(True, alpha=0.3)

# Add median and quartile labels for Dataset 1 separated box plot
for i, (data, label) in enumerate(zip([non_corrupted1, corrupted1], ['Non-corrupted', 'Corrupted']), 1):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    median = data.median()
    
    ax4.text(i, median, f'Median:\n{median:.3f}', 
             ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    ax4.text(i, q3, f'Q3:\n{q3:.3f}', 
             ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    ax4.text(i, q1, f'Q1:\n{q1:.3f}', 
             ha='center', va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Dataset 1 box plot (combined)
ax5 = axes[1, 1]
ax5.boxplot([all_scores1], tick_labels=['All Examples'])
ax5.set_ylabel('Similarity Score')
ax5.set_title('Dataset 1: Combined')
ax5.grid(True, alpha=0.3)

# Add median and quartile labels for Dataset 1 combined box plot
q1_comb = all_scores1.quantile(0.25)
q3_comb = all_scores1.quantile(0.75)
median_comb = all_scores1.median()

ax5.text(1, median_comb, f'Median:\n{median_comb:.3f}', 
         ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
ax5.text(1, q3_comb, f'Q3:\n{q3_comb:.3f}', 
         ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
ax5.text(1, q1_comb, f'Q1:\n{q1_comb:.3f}', 
         ha='center', va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Dataset 2 box plot
ax6 = axes[1, 2]
ax6.boxplot([similarities2], tick_labels=['All'])
ax6.set_ylabel('Similarity Score')
ax6.set_title('Dataset 2: Distribution')
ax6.grid(True, alpha=0.3)

# Add median and quartile labels for Dataset 2 box plot
q1_d2 = similarities2.quantile(0.25)
q3_d2 = similarities2.quantile(0.75)
median_d2 = similarities2.median()

ax6.text(1, median_d2, f'Median:\n{median_d2:.3f}', 
         ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
ax6.text(1, q3_d2, f'Q3:\n{q3_d2:.3f}', 
         ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
ax6.text(1, q1_d2, f'Q1:\n{q1_d2:.3f}', 
         ha='center', va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
output_path = os.path.join(script_dir, 'both_datasets_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved comparison plot to {output_path}")

# Show top and bottom examples from dataset 2
print("\n" + "="*80)
print("TOP 5 SIMILARITY SCORES FROM bird_sql2nl_cot_dataset:")
print("="*80)
top2 = df2.nlargest(5, similarity_col)
for idx, row in top2.iterrows():
    print(f"\n--- Similarity: {row[similarity_col]:.4f} ---")
    print(f"SQL: {str(row['sql'])[:150]}...")
    print(f"Predicted NL: {row['predicted_nl']}")
    print(f"True NL: {row['true_nl']}")

print("\n" + "="*80)
print("BOTTOM 5 SIMILARITY SCORES FROM bird_sql2nl_cot_dataset:")
print("="*80)
bottom2 = df2.nsmallest(5, similarity_col)
for idx, row in bottom2.iterrows():
    print(f"\n--- Similarity: {row[similarity_col]:.4f} ---")
    print(f"SQL: {str(row['sql'])[:150]}...")
    print(f"Predicted NL: {row['predicted_nl']}")
    print(f"True NL: {row['true_nl']}")

