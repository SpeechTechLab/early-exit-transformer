import os
import glob

features_dir = "features_103_QCP"  # Directory with your CSV features
manifest_file = "100h_manifest.txt"

# Find all .trans.txt files recursively
trans_files = glob.glob("LibriSpeech/train-clean-100/*/*/*.trans.txt")

# Load all transcriptions into a dict
trans_dict = {}
for trans_file in trans_files:
    with open(trans_file, "r") as f:
        for line in f:
            utt_id, *trans = line.strip().split()
            trans_dict[utt_id] = " ".join(trans)

with open(manifest_file, "w") as out_f:
    for fname in os.listdir(features_dir):
        if fname.endswith(".csv"):
            utt_id = fname.replace(".csv", "")
            if utt_id in trans_dict:
                feat_path = os.path.join(features_dir, fname)
                out_f.write(f"{feat_path},{trans_dict[utt_id]}\n")
            else:
                print(f"Warning: No transcription for {utt_id}")

print(f"Manifest written to {manifest_file}")
