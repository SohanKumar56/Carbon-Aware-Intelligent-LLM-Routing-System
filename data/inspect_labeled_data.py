"""Quick inspection of the labeled dataset"""
import pandas as pd
from pathlib import Path

labeled_dir = Path(__file__).parent / "labeled"
df = pd.read_parquet(labeled_dir / "prompt_complexity_labeled.parquet")

print("="*80)
print("LABELED DATASET INSPECTION")
print("="*80)
print(f"\nTotal samples: {len(df):,}")
print(f"\nColumns: {list(df.columns)}")

print("\n" + "="*80)
print("CLASS DISTRIBUTION")
print("="*80)
class_dist = df['complexity_label'].value_counts()
for label, count in class_dist.items():
    pct = 100 * count / len(df)
    print(f"{label:8s}: {count:6,} ({pct:5.1f}%)")

print("\n" + "="*80)
print("SOURCE DISTRIBUTION BY CLASS")
print("="*80)
crosstab = pd.crosstab(df['source_dataset'], df['complexity_label'], margins=True)
print(crosstab)

print("\n" + "="*80)
print("SAMPLE PROMPTS BY CLASS")
print("="*80)

for label in ['small', 'medium', 'large']:
    print(f"\n--- {label.upper()} Examples ---")
    samples = df[df['complexity_label'] == label].sample(n=min(3, len(df[df['complexity_label'] == label])), random_state=42)
    for i, row in enumerate(samples.iterrows(), 1):
        _, data = row
        source = data['source_dataset']
        prompt = data['prompt']
        prompt_preview = prompt[:150] + "..." if len(prompt) > 150 else prompt
        print(f"\n{i}. [{source}]")
        print(f"   {prompt_preview}")

print("\n" + "="*80)
