import pandas as pd
import json

# 1. Count Authentic Lazada Data (added 'r' before the string to fix the path error)
df_qa = pd.read_csv(r'D:\thesis_prototype\data\01_raw\lazada-qa-taglish.csv')
with open(r'D:\thesis_prototype\data\01_raw\lazada-review-filipino.json', 'r', encoding='utf-8') as f:
    reviews = json.load(f)
    
authentic_count = len(df_qa) + len(reviews)

# 2. Count Synthetic Data (added 'r' before the string)
df_synth = pd.read_csv(r'D:\thesis_prototype\data\01_raw\raw_synthetic_data.csv')
synthetic_count = len(df_synth)

ratio = synthetic_count / authentic_count

print(f"Exact Authentic Count: {authentic_count}")
print(f"Exact Synthetic Count: {synthetic_count}")
print(f"Exact Ratio: 1 authentic sample to {ratio:.2f} synthetic samples")