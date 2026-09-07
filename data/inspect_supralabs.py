"""Quick inspection of SupraLabs dataset labels"""
import pandas as pd
from pathlib import Path

cache_dir = Path(__file__).parent / "cache"
df = pd.read_parquet(cache_dir / "supralabs_routing.parquet")

print("="*80)
print("SupraLabs Routing Choice Distribution")
print("="*80)
print(df['routing_choice'].value_counts())
print()

print("="*80)
print("Complexity Score Distribution")
print("="*80)
print(df['complexity_score'].describe())
print()

print("="*80)
print("Sample rows with routing choices and scores")
print("="*80)
for i, row in df.head(15).iterrows():
    prompt_preview = row['prompt'][:80] + "..." if len(row['prompt']) > 80 else row['prompt']
    print(f"{i+1}. [{row['routing_choice']}] Score: {row['complexity_score']:.2f}")
    print(f"   {prompt_preview}")
    print()
