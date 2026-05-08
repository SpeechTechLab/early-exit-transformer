import torch
import numpy as np
import os

from util.data_loader import CollatePaddingFn, CollateInferFn

# ---- Audio backend safety (fix TorchCodec/FFmpeg issues) ----
# Newer torchaudio versions may default to TorchCodec, which requires FFmpeg
# shared libraries. In minimal/docker environments this often fails at runtime.
#
# We force torchaudio to avoid TorchCodec and prefer classic backends.
os.environ.setdefault("TORCHAUDIO_USE_TORCHCODEC", "0")

import torchaudio

try:
    # Prefer sox_io when available; otherwise soundfile is fine.
    # (If neither is available, torchaudio will fall back to its defaults.)
    torchaudio.set_audio_backend("sox_io")
except Exception:
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass


class PrecomputedFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path):
        self.entries = []

        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Manifest format: feature_csv_path,transcription text...
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue

                feat_path, text = parts
                utt_id = os.path.splitext(os.path.basename(feat_path))[0]
                self.entries.append((feat_path, text, utt_id))

        if len(self.entries) == 0:
            raise ValueError(f"No valid entries found in manifest: {manifest_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        feat_path, text, utt_id = self.entries[idx]
        arr = np.loadtxt(feat_path, delimiter=",")

        # Return shape [feature_dim, time] to match downstream batching expectations.
        if arr.ndim == 1:
            feat = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
        else:
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T
            feat = torch.tensor(arr, dtype=torch.float32)

        return feat, 16000, text, "spk", utt_id


def _librispeech_root(args):
    """torchaudio LIBRISPEECH(root, url=...) expects root to contain LibriSpeech/<split>."""
    return getattr(args, "librispeech_root", None) or ""


def get_data_loader(args):
    if getattr(args, "use_precomputed_features", False):
        if not args.manifest:
            raise ValueError("--manifest is required when --use_precomputed_features is set")
        train_dataset = PrecomputedFeatureDataset(args.manifest)
    else:
        split = getattr(args, "train_split", "all")
        ls_root = _librispeech_root(args)
        try:
            train_dataset1 = torchaudio.datasets.LIBRISPEECH(
                ls_root, url="train-clean-100", download=False
            )

            if split in ("100h", "train-clean-100"):
                train_dataset = train_dataset1
            else:
                train_dataset2 = torchaudio.datasets.LIBRISPEECH(
                    ls_root, url="train-clean-360", download=False
                )
                train_dataset3 = torchaudio.datasets.LIBRISPEECH(
                    ls_root, url="train-other-500", download=False
                )
                train_dataset = torch.utils.data.ConcatDataset(
                    [train_dataset1, train_dataset2, train_dataset3]
                )
        except RuntimeError as e:
            if "Dataset not found" not in str(e):
                raise
            print("LibriSpeech training set not found locally. Downloading required split(s)...")
            train_dataset1 = torchaudio.datasets.LIBRISPEECH(
                ls_root, url="train-clean-100", download=True
            )

            if split in ("100h", "train-clean-100"):
                train_dataset = train_dataset1
            else:
                train_dataset2 = torchaudio.datasets.LIBRISPEECH(
                    ls_root, url="train-clean-360", download=True
                )
                train_dataset3 = torchaudio.datasets.LIBRISPEECH(
                    ls_root, url="train-other-500", download=True
                )
                train_dataset = torch.utils.data.ConcatDataset(
                    [train_dataset1, train_dataset2, train_dataset3]
                )

    # Optional debug/overfit mode: subset to a small number of utterances.
    max_utts = int(getattr(args, "max_train_utts", 0) or 0)
    if max_utts > 0 and len(train_dataset) > max_utts:
        subset_seed = int(getattr(args, "subset_seed", 0) or 0)
        if subset_seed != 0:
            g = torch.Generator()
            g.manual_seed(subset_seed)
            idx = torch.randperm(len(train_dataset), generator=g)[:max_utts].tolist()
        else:
            idx = list(range(max_utts))
        train_dataset = torch.utils.data.Subset(train_dataset, idx)
        print(f"INFO: training subset enabled: {len(train_dataset)} utterances")

    collate_padding_fn = CollatePaddingFn(args=args)
    data_loader = torch.utils.data.DataLoader(train_dataset, 
                                              pin_memory=False, 
                                              batch_size=args.batch_size,
                                              shuffle=args.shuffle, 
                                              collate_fn=collate_padding_fn, 
                                              num_workers=args.n_workers)
    # data_loader_initial = torch.utils.data.DataLoader(
    # train_dataset1, pin_memory=False, batch_size=args.batch_size, shuffle=args.shuffle, collate_fn=collate_padding_fn, num_workers=args.n_workers)

    return data_loader


def get_infer_data_loader(args, split=None, shuffle=None):

    if shuffle == None:
        shuffle = args.shuffle

    ls_root = _librispeech_root(args)
    try:
        try:
            train_dataset = torchaudio.datasets.LIBRISPEECH(
                ls_root, url=split, download=False
            )
        except RuntimeError as e:
            if "Dataset not found" not in str(e):
                raise
            print(f"LibriSpeech {split} not found locally. Downloading dataset...")
            train_dataset = torchaudio.datasets.LIBRISPEECH(
                ls_root, url=split, download=True
            )

        collate_infer_fn = CollateInferFn(args=args)
        # Optional debug mode: subset inference to small number of utterances
        max_utts = int(getattr(args, "max_infer_utts", 0) or 0)
        if max_utts > 0 and len(train_dataset) > max_utts:
            subset_seed = int(getattr(args, "subset_seed", 0) or 0)
            if subset_seed != 0:
                g = torch.Generator()
                g.manual_seed(subset_seed)
                idx = torch.randperm(len(train_dataset), generator=g)[:max_utts].tolist()
            else:
                idx = list(range(max_utts))
            train_dataset = torch.utils.data.Subset(train_dataset, idx)
            print(f"INFO: inference subset enabled: {len(train_dataset)} utterances")

        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            pin_memory=False,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate_infer_fn,
            num_workers=args.n_workers,
        )
        return data_loader

    except Exception as e:
        import sys
        import traceback

        print(
            f"ERROR: could not load LibriSpeech split {split!r} under root {ls_root!r} "
            f"(expect {ls_root or '.'}/LibriSpeech/{split}).",
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(f"Invalid data split ({split!r}): {e}")