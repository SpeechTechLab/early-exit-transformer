import os
import pandas as pd

# Paths
input_csv = "features_103_QCP.csv"
out_dir = "features_103_QCP"
os.makedirs(out_dir, exist_ok=True)

# Read the big CSV
print("Reading CSV...")
df = pd.read_csv(input_csv)

# Drop non-feature columns for output (keep only features)
# We'll keep all columns except: file_name, speaker, chapter, utterance, label, task
non_feature_cols = ["file_name", "speaker", "chapter", "utterance", "label", "task"]
feature_cols = [col for col in df.columns if col not in non_feature_cols]

print(f"Splitting {len(df)} rows...")
for idx, row in df.iterrows():
    utt_id = row["file_name"]
    feature_row = row[feature_cols]
    out_path = os.path.join(out_dir, f"{utt_id}.csv")
    feature_row.to_frame().T.to_csv(out_path, index=False)

print(f"Done! Wrote per-utterance CSVs to {out_dir}/")
