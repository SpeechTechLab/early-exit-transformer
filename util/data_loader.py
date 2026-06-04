import re
import os
import csv
from pathlib import Path
import torch
import torchaudio.transforms as T
import torch.nn.functional as F

# For loading precomputed features
import numpy as np


def _normalize_utt_id(raw_id):
    base = os.path.basename(str(raw_id))
    return os.path.splitext(base)[0]


def normalize_label_for_bpe(label: str, *, uppercase: bool = True) -> str:
    """Normalize transcript text for SentencePiece BPE.

    LibriSpeech BPE (``libri.bpe-256.model``) expects UPPERCASE.
    SpeechTek English-EE BPE (``bpe-256.model`` on HuggingFace) expects lowercase.
    """
    text = str(label)
    if uppercase:
        text = text.upper()
        text = re.sub(r"[^A-Z0-9' ]+", " ", text)
    else:
        text = text.lower()
        text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_label_for_bpe_args(args, label: str) -> str:
    return normalize_label_for_bpe(label, uppercase=getattr(args, "bpe_uppercase", True))


def _min_mel_frames(args) -> int:
    """Skip utterances shorter than this (mel bins) to avoid Conformer BN errors on T=1."""
    return int(getattr(args, "min_mel_frames", 100))


def _max_mel_frames(max_len: int, model_type: str = "") -> int:
    """Max mel time steps so Zipformer/Conformer subsampling stays within max_len PE."""
    max_len = int(max_len)
    if "zipformer" in str(model_type).lower():
        # Conv1d k=3 s=2: enc_len = (mel - 3) // 2 + 1
        return 2 * max_len + 1
    # Conformer-style two stride-2 convs: enc_len ≈ mel // 4
    return 4 * max_len + 8


def _truncate_feature_time(spec, args, ut_id=None):
    """Truncate [feat, time] tensors to fit positional encoding (args.max_len)."""
    if spec.dim() != 2:
        return spec
    max_frames = _max_mel_frames(
        getattr(args, "max_len", 2000),
        getattr(args, "model_type", ""),
    )
    if spec.size(1) > max_frames:
        warned = getattr(args, "_mel_trunc_warn_count", 0)
        if warned < 5:
            print(
                f"INFO: truncating {ut_id or 'utterance'} mel frames "
                f"{spec.size(1)} -> {max_frames} (max_len={args.max_len})"
            )
            args._mel_trunc_warn_count = warned + 1
        spec = spec[:, :max_frames]
    return spec


def _extract_utt_id(sample):
    """Return utterance ID from dataset sample.

    torchaudio LIBRISPEECH samples are:
    (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)

    Precomputed samples in this repo are:
    (features, sample_rate, text, speaker_id, utt_id)
    """
    if len(sample) >= 6:
        # torchaudio LIBRISPEECH provides (speaker_id, chapter_id, utterance_id)
        # as numeric fields; glottal CSV keys use speaker-chapter-utterance.
        speaker_id = sample[3]
        chapter_id = sample[4]
        utterance_id = sample[5]

        utt_str = str(utterance_id)
        if "-" in utt_str:
            return utt_str

        # Keep common LibriSpeech format: <speaker>-<chapter>-<utt:04d>
        try:
            utt_str = f"{int(utterance_id):04d}"
        except Exception:
            utt_str = utt_str.zfill(4)

        return f"{speaker_id}-{chapter_id}-{utt_str}"
    if len(sample) >= 5:
        return sample[4]
    return "unknown"


def _select_numeric_feature_columns(fieldnames, drop_mfcc):
    meta_cols = {
        "file_name", "speaker", "label", "task", "utt_id", "ut_id", "id", "path",
        "chapter_id", "transcript", "text", "sentence",
        "frame_idx", "frame_start_sample", "frame_start_sec", "voiced", "f0"
    }
    out = []
    for col in fieldnames:
        c = col.strip()
        cl = c.lower()
        if cl in meta_cols:
            continue
        if drop_mfcc and cl.startswith("mfcc_"):
            continue
        out.append(c)
    return out


def _compute_standardization_stats(feature_matrix, eps=1e-8):
    mean = np.mean(feature_matrix, axis=0).astype(np.float32)
    std = np.std(feature_matrix, axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return mean, std


def _load_standardization_stats(stats_path, feat_dim):
    data = np.load(stats_path)
    if "mean" not in data or "std" not in data:
        raise ValueError(
            f"Invalid glottal stats file (expected keys 'mean' and 'std'): {stats_path}"
        )
    mean = np.asarray(data["mean"], dtype=np.float32).ravel()
    std = np.asarray(data["std"], dtype=np.float32).ravel()
    if mean.size != feat_dim or std.size != feat_dim:
        raise ValueError(
            f"Glottal stats dimension mismatch in {stats_path}: expected {feat_dim}, "
            f"got mean={mean.size}, std={std.size}"
        )
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def _apply_standardization(feature_matrix, mean, std):
    return (feature_matrix - mean) / std


def load_glottal_feature_map(
    csv_path,
    drop_mfcc=True,
    standardize=True,
    stats_in_path=None,
    stats_out_path=None,
    utt_level_mean=False,
):
    feat_dim = None
    id_candidates = ["file_name", "utt_id", "ut_id", "id", "path"]

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Invalid CSV (no header): {csv_path}")

        cols = _select_numeric_feature_columns(reader.fieldnames, drop_mfcc)
        frame_level = any(str(c).strip().lower() == "frame_idx" for c in reader.fieldnames)

        if not frame_level:
            per_utt_values = {}
            for row in reader:
                row_id = None
                for key in id_candidates:
                    if key in row and row[key] is not None and str(row[key]).strip() != "":
                        row_id = row[key]
                        break
                if row_id is None:
                    continue

                values = []
                for col in cols:
                    v = row.get(col, "")
                    try:
                        x = float(v)
                    except (TypeError, ValueError):
                        x = 0.0
                    if np.isnan(x) or np.isinf(x):
                        x = 0.0
                    values.append(x)

                if len(values) == 0:
                    continue

                if feat_dim is None:
                    feat_dim = len(values)
                elif len(values) != feat_dim:
                    raise ValueError(
                        f"Inconsistent glottal feature dimension in {csv_path}: expected {feat_dim}, got {len(values)}"
                    )

                uid = _normalize_utt_id(row_id)
                per_utt_values.setdefault(uid, []).append(np.asarray(values, dtype=np.float32))

            if feat_dim is None or len(per_utt_values) == 0:
                raise ValueError(f"No usable glottal features found in CSV: {csv_path}")

            utt_ids = sorted(per_utt_values.keys())
            per_utt_matrix = []
            duplicate_rows = 0

            for uid in utt_ids:
                stacked = np.vstack(per_utt_values[uid])
                if stacked.shape[0] > 1:
                    duplicate_rows += stacked.shape[0] - 1
                # CSV can contain multiple rows per utterance; average them into one vector.
                per_utt_matrix.append(np.mean(stacked, axis=0))

            feature_matrix = np.vstack(per_utt_matrix).astype(np.float32)
            if standardize:
                if stats_in_path:
                    mean, std = _load_standardization_stats(stats_in_path, feat_dim)
                    print(f"INFO: loaded glottal normalization stats from {stats_in_path}")
                else:
                    mean, std = _compute_standardization_stats(feature_matrix)

                if stats_out_path:
                    stats_out = Path(stats_out_path)
                    stats_out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(stats_out, mean=mean, std=std)
                    print(f"INFO: saved glottal normalization stats to {stats_out}")

                feature_matrix = _apply_standardization(feature_matrix, mean, std)

            feat_map = {
                uid: torch.tensor(feature_matrix[idx], dtype=torch.float32)
                for idx, uid in enumerate(utt_ids)
            }

            if duplicate_rows > 0:
                print(
                    f"INFO: aggregated {duplicate_rows} duplicate glottal CSV rows into per-utterance means"
                )
            if standardize:
                print("INFO: standardized glottal features with per-dimension z-score")

            return feat_map, feat_dim

        print("INFO: loading frame-level glottal CSV (streaming mode). This can take several minutes...")

        per_utt_matrix = {}
        duplicate_rows = 0
        processed_rows = 0

        current_uid = None
        current_frame_idxs = []
        current_vecs = []

        def flush_current_block(uid, frame_idxs, vecs):
            nonlocal duplicate_rows

            if uid is None or len(vecs) == 0:
                return

            idx_arr = np.asarray(frame_idxs, dtype=np.int64)
            val_arr = np.vstack(vecs).astype(np.float32, copy=False)

            if feat_dim is not None and val_arr.shape[1] != feat_dim:
                raise ValueError(
                    f"Inconsistent frame-level glottal feature dimension in {csv_path}: "
                    f"expected {feat_dim}, got {val_arr.shape[1]}"
                )

            order = np.argsort(idx_arr, kind="stable")
            idx_arr = idx_arr[order]
            val_arr = val_arr[order]

            uniq, inverse, counts = np.unique(idx_arr, return_inverse=True, return_counts=True)
            if np.any(counts > 1):
                duplicate_rows += int(np.sum(counts - 1))
                agg = np.zeros((uniq.size, val_arr.shape[1]), dtype=np.float32)
                np.add.at(agg, inverse, val_arr)
                agg /= counts[:, None].astype(np.float32)
                block = agg
            else:
                block = val_arr

            if uid in per_utt_matrix:
                per_utt_matrix[uid] = np.vstack([per_utt_matrix[uid], block])
            else:
                per_utt_matrix[uid] = block

        for row in reader:
            row_id = None
            for key in id_candidates:
                if key in row and row[key] is not None and str(row[key]).strip() != "":
                    row_id = row[key]
                    break
            if row_id is None:
                continue

            values = []
            for col in cols:
                v = row.get(col, "")
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    x = 0.0
                if np.isnan(x) or np.isinf(x):
                    x = 0.0
                values.append(x)

            if len(values) == 0:
                continue

            if feat_dim is None:
                feat_dim = len(values)
            elif len(values) != feat_dim:
                raise ValueError(
                    f"Inconsistent frame-level glottal feature dimension in {csv_path}: expected {feat_dim}, got {len(values)}"
                )

            uid = _normalize_utt_id(row_id)
            raw_idx = row.get("frame_idx", "")
            try:
                frame_idx = int(float(raw_idx))
            except (TypeError, ValueError):
                frame_idx = len(current_frame_idxs)

            if current_uid is None:
                current_uid = uid

            if uid != current_uid:
                flush_current_block(current_uid, current_frame_idxs, current_vecs)
                current_uid = uid
                current_frame_idxs = []
                current_vecs = []

            current_frame_idxs.append(frame_idx)
            current_vecs.append(np.asarray(values, dtype=np.float32))

            processed_rows += 1
            if processed_rows % 1000000 == 0:
                print(f"INFO: parsed {processed_rows:,} frame rows from {csv_path}")

        flush_current_block(current_uid, current_frame_idxs, current_vecs)

    if feat_dim is None or len(per_utt_matrix) == 0:
        raise ValueError(f"No usable frame-level glottal features found in CSV: {csv_path}")

    utt_ids = sorted(per_utt_matrix.keys())

    if standardize:
        if stats_in_path:
            mean, std = _load_standardization_stats(stats_in_path, feat_dim)
            print(f"INFO: loaded glottal normalization stats from {stats_in_path}")
        else:
            total_rows = 0
            sum_vec = np.zeros(feat_dim, dtype=np.float64)
            sqsum_vec = np.zeros(feat_dim, dtype=np.float64)
            for uid in utt_ids:
                mat = per_utt_matrix[uid]
                total_rows += mat.shape[0]
                sum_vec += np.sum(mat, axis=0, dtype=np.float64)
                sqsum_vec += np.sum(np.square(mat, dtype=np.float64), axis=0, dtype=np.float64)

            mean = (sum_vec / max(total_rows, 1)).astype(np.float32)
            var = (sqsum_vec / max(total_rows, 1)) - np.square(mean.astype(np.float64))
            var = np.maximum(var, 0.0)
            std = np.sqrt(var).astype(np.float32)
            std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

        if stats_out_path:
            stats_out = Path(stats_out_path)
            stats_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(stats_out, mean=mean, std=std)
            print(f"INFO: saved glottal normalization stats to {stats_out}")

        for uid in utt_ids:
            per_utt_matrix[uid] = _apply_standardization(per_utt_matrix[uid], mean, std).astype(np.float32)

    if utt_level_mean:
        # Collapse frame-level matrix -> utterance-level vector.
        # This enables "one glottal vector per utterance", later repeated across time.
        for uid in utt_ids:
            per_utt_matrix[uid] = np.mean(per_utt_matrix[uid], axis=0, dtype=np.float32)
        feat_map = {uid: torch.tensor(per_utt_matrix[uid], dtype=torch.float32) for uid in utt_ids}
    else:
        feat_map = {uid: torch.tensor(per_utt_matrix[uid].T, dtype=torch.float32) for uid in utt_ids}

    if duplicate_rows > 0:
        print(f"INFO: aggregated {duplicate_rows} duplicate frame rows in glottal CSV")
    print(
        f"INFO: loaded {'utterance-mean' if utt_level_mean else 'frame-level'} glottal features "
        f"for {len(feat_map)} utterances from {processed_rows:,} rows"
    )
    if standardize:
        print("INFO: standardized frame-level glottal features with per-dimension z-score")

    return feat_map, feat_dim


def infer_glottal_feature_dim(csv_path, drop_mfcc=True):
    _, feat_dim = load_glottal_feature_map(csv_path, drop_mfcc=drop_mfcc)
    return feat_dim


def compute_glottal_features_from_waveform(
    waveform,
    sample_rate,
    mel_time_steps,
    args,
    *,
    norm_mean=None,
    norm_std=None,
):
    """Compute QCP glottal frames from the same raw waveform used for mels, then align to mel time steps.

    Framing uses a 50 ms analysis window and ``args.hop_length`` as frame shift so the QCP frame grid
    tracks the STFT hop used for spectrograms; lengths are matched to ``mel_time_steps`` with linear
    interpolation (same idea as the CSV glottal branch).

    ``waveform`` is expected as ``[1, T]`` or ``[T]``. ``norm_mean`` / ``norm_std`` are optional
    per-channel vectors (length ``QCP_FRAME_FEATURE_DIM``), e.g. from ``glottal_norm_stats_in``.
    """
    from extract_glottal_features_qcp import (
        QCP_FRAME_FEATURE_DIM,
        glottal_frame_tensor_from_waveform_numpy,
    )

    device = waveform.device
    w = waveform.detach().float().cpu()
    if w.dim() == 2:
        w = w.mean(dim=0)
    else:
        w = w.reshape(-1)

    sr = int(sample_rate)
    hop = int(getattr(args, "hop_length", max(1, int(round(0.01 * sr)))))
    frame_length = int(round(0.050 * sr))

    g = glottal_frame_tensor_from_waveform_numpy(
        w.numpy().astype(np.float64),
        sr,
        frame_length,
        hop,
        preprocess=True,
    )

    if mel_time_steps <= 0:
        return torch.zeros(QCP_FRAME_FEATURE_DIM, 0, dtype=torch.float32, device=device)

    if g.size(1) == 0:
        g = torch.zeros(QCP_FRAME_FEATURE_DIM, mel_time_steps, dtype=torch.float32)
    elif g.size(1) != mel_time_steps:
        g = F.interpolate(
            g.unsqueeze(0),
            size=mel_time_steps,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

    if norm_mean is not None and norm_std is not None:
        m = torch.as_tensor(norm_mean, dtype=torch.float32, device=g.device).view(-1, 1)
        s = torch.as_tensor(norm_std, dtype=torch.float32, device=g.device).view(-1, 1)
        g = (g - m) / s

    return g.to(device=device, dtype=torch.float32)


def spec_transform(waveform, args):
    spec_t = T.Spectrogram(n_fft=args.n_fft * 2,
                           hop_length=args.hop_length,
                           win_length=args.win_length)
    return spec_t(waveform)


def melspec_transform(waveform, args):
    melspec_t = T.MelScale(sample_rate=args.sample_rate,
                           n_mels=args.n_mels,
                           n_stft=args.n_fft+1)
    return melspec_t(waveform)


def pad_sequence(batch, padvalue):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=padvalue)
    return batch.permute(0, 2, 1)


class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        # 30
        ^ 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        ' 29
        $ 31
        @ 0
        """
        # ^=<SOS> 1
        # $=<EOS> 31
        # #=<PAD> 30
        # @=<blank> for ctc
        char_map = {}
        index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            char_map[ch] = int(index)
            index_map[int(index)] = ch
        index_map[28] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = 28  # char_map['']
            else:
                ch = char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(index_map[i.detach().item()])
        return ''.join(string)  # .replace('', ' ')


text_transform = TextTransform()


class CollateFn(object):

    def __init__(self, args):
        self.args = args

    def load_features_csv(self, path):
        features = np.loadtxt(path, delimiter=',')
        return torch.tensor(features, dtype=torch.float32)

    def __call__(self, batch,
                 SOS_token=None, EOS_token=None, PAD_token=None):

        if SOS_token == None:
            SOS_token = self.args.trg_sos_idx
        if EOS_token == None:
            EOS_token = self.args.trg_eos_idx
        if PAD_token == None:
            PAD_token = self.args.trg_pad_idx

        tensors, targets = [], []
        t_len = []
        t_source = []
        k = 0
        # Gather in lists, and encode labels as indices
        for waveform, smp_freq, label, spk_id, ut_id, *rest in batch:
            # If using precomputed features, waveform is actually the path to the feature file
            if hasattr(self.args, 'use_precomputed_features') and self.args.use_precomputed_features:
                spec = self.load_features_csv(waveform)
                spec = spec.unsqueeze(0) if spec.dim() == 2 else spec
                t_source += [spec.size(2) if spec.dim() == 3 else spec.size(1)]
                tensors += spec
                del spec
            else:
                label = re.sub(r"<unk>|\[ unclear \]", "", label)
                label = re.sub(r"[#^$?:;.!\[\]]+", "", label)
                if len(label) < self.args.max_utterance_length:
                    if not (hasattr(self.args, 'use_precomputed_features') and self.args.use_precomputed_features):
                        # Only do this if not using precomputed features
                        pass  # ...existing code for audio feature extraction...
                    if self.args.bpe == True:
                        bpe_label = normalize_label_for_bpe_args(self.args, label)
                        if not bpe_label:
                            print('REMOVED:', ut_id, ' LAB: (empty after BPE normalize)')
                            continue
                        tg = torch.LongTensor(
                            [self.args.sp.bos_id()]
                            + self.args.sp.encode_as_ids(bpe_label)
                            + [self.args.sp.eos_id()]
                        )
                    else:
                        tg = torch.LongTensor(
                            text_transform.text_to_int("^"+label.lower()+"$"))
                    targets += [tg.unsqueeze(0)]
                    t_len += [len(tg)]
                    k = k+1
                    del waveform
                    del label
                else:
                    print('REMOVED:', ut_id, ' LAB:', label)

        if tensors:
            tensors = pad_sequence(tensors, 0)
            targets = pad_sequence(targets, PAD_token)
            return tensors.squeeze(1), targets.squeeze(1), torch.tensor(t_len), torch.tensor(t_source)
        else:
            return None


class CollatePaddingFn(object):
    def __init__(self, args):
        self.args = args
        self.glottal_feat_map = None
        self.glottal_dim = 0
        self._glottal_norm_mean = None
        self._glottal_norm_std = None
        self._missing_glottal_warned = False
        # glottal_missing_count: utterances that used a zero glottal tensor (CSV id not in map).
        # glottal_found_count: CSV hit, or every utterance when glottal_from_waveform.
        self.glottal_missing_count = 0
        self.glottal_found_count = 0

        if getattr(args, "append_glottal_features", False):
            if getattr(args, "glottal_from_waveform", False):
                from extract_glottal_features_qcp import QCP_FRAME_FEATURE_DIM

                self.glottal_dim = QCP_FRAME_FEATURE_DIM
                self.glottal_feat_map = None
                stats_in = getattr(args, "glottal_norm_stats_in", None)
                if getattr(args, "glottal_standardize", True) and stats_in:
                    self._glottal_norm_mean, self._glottal_norm_std = _load_standardization_stats(
                        stats_in, self.glottal_dim
                    )
            else:
                if not args.glottal_features_path:
                    raise ValueError(
                        "--glottal_features_path is required when --append_glottal_features is set "
                        "(unless --glottal_from_waveform is set)"
                    )
                self.glottal_feat_map, self.glottal_dim = load_glottal_feature_map(
                    args.glottal_features_path,
                    drop_mfcc=getattr(args, "glottal_drop_mfcc", False),
                    standardize=getattr(args, "glottal_standardize", True),
                    stats_in_path=getattr(args, "glottal_norm_stats_in", None),
                    stats_out_path=getattr(args, "glottal_norm_stats_out", None),
                    utt_level_mean=getattr(args, "glottal_utt_mean", False),
                )

    def __call__(self, batch,
                 SOS_token=None, EOS_token=None, PAD_token=None):
        if SOS_token == None:
            SOS_token = self.args.trg_sos_idx
        if EOS_token == None:
            EOS_token = self.args.trg_eos_idx
        if PAD_token == None:
            PAD_token = self.args.trg_pad_idx

        # Gather in lists, and encode labels as indices
        batch = sorted(batch, key=lambda x: x[0].size(1), reverse=True)

        n_split = self.args.n_batch_split
        s_sum = sum(x[0].size(1) for x in batch) / n_split
        p_sum = 0
        chunked_batch = list()
        init = 0
        end = 0
        p_split = 0

        for w, *_ in batch:
            p_sum += w.size(1)

            if p_sum >= s_sum:
                chunked_batch.append(batch[init:end+1])
                p_sum = 0
                p_split += 1
                init = end+1

            end += 1

        if p_split != n_split:
            chunked_batch.append(batch[init:end])

        out_batch = []
        for c_batch in chunked_batch:
            tensors, targets, t_len, t_source, o_batch = [], [], [], [], []
            k = 0

            for sample in c_batch:
                waveform, smp_freq, label, spk_id = sample[:4]
                ut_id = _extract_utt_id(sample)
                smp_freq = int(smp_freq)
                label = re.sub(r"<unk>|\[ unclear \]", "", label)
                label = re.sub(r"[#^$?:;.!\[\]]+", "", label)

                if len(label) < self.args.max_utterance_length:
                    if getattr(self.args, "use_precomputed_features", False):
                        # Precomputed tensors are expected as [feature_dim, time].
                        spec = waveform.float()
                        if spec.dim() == 1:
                            spec = spec.unsqueeze(1)
                        spec = _truncate_feature_time(spec, self.args, ut_id)
                    else:
                        spec = spec_transform(waveform, self.args)  # .to(device)
                        spec = melspec_transform(spec, self.args)

                    if spec.dim() == 3:
                        spec = spec.squeeze(0)
                    spec = _truncate_feature_time(spec, self.args, ut_id)
                    if spec.size(1) < _min_mel_frames(self.args):
                        print(
                            f"REMOVED: {ut_id}  (too short: {spec.size(1)} mel frames, "
                            f"min {_min_mel_frames(self.args)})"
                        )
                        continue

                    if getattr(self.args, "append_glottal_features", False):
                        if getattr(self.args, "glottal_from_waveform", False):
                            g_rep = compute_glottal_features_from_waveform(
                                waveform,
                                smp_freq,
                                spec.size(1),
                                self.args,
                                norm_mean=self._glottal_norm_mean,
                                norm_std=self._glottal_norm_std,
                            )
                            g_rep = g_rep.to(spec.device, dtype=spec.dtype)
                            self.glottal_found_count += 1
                        else:
                            uid = _normalize_utt_id(ut_id)
                            g = self.glottal_feat_map.get(uid)
                            if g is None:
                                g_rep = torch.zeros(self.glottal_dim, spec.size(1), dtype=torch.float32)
                                self.glottal_missing_count += 1
                                if not self._missing_glottal_warned:
                                    print(f"WARNING: missing glottal features for utterance '{uid}'. Using zeros.")
                                    self._missing_glottal_warned = True
                            else:
                                self.glottal_found_count += 1
                                g = g.to(spec.device)
                                if g.dim() == 1:
                                    g_rep = g.unsqueeze(1).repeat(1, spec.size(1))
                                elif g.dim() == 2:
                                    if g.size(1) == spec.size(1):
                                        g_rep = g
                                    else:
                                        g_rep = F.interpolate(
                                            g.unsqueeze(0),
                                            size=spec.size(1),
                                            mode="linear",
                                            align_corners=False,
                                        ).squeeze(0)
                                else:
                                    g = g.reshape(self.glottal_dim, -1)
                                    g_rep = F.interpolate(
                                        g.unsqueeze(0),
                                        size=spec.size(1),
                                        mode="linear",
                                        align_corners=False,
                                    ).squeeze(0)
                        spec = torch.cat([spec, g_rep], dim=0)

                    if self.args.bpe == True:
                        bpe_label = normalize_label_for_bpe_args(self.args, label)
                        if not bpe_label:
                            print('REMOVED:', ut_id, ' LAB: (empty after BPE normalize)')
                            continue
                        tg = torch.LongTensor(
                            [self.args.sp.bos_id()]
                            + self.args.sp.encode_as_ids(bpe_label)
                            + [self.args.sp.eos_id()]
                        )
                    else:
                        tg = torch.LongTensor(
                            text_transform.text_to_int("^"+label.lower()+"$"))

                    if spec.dim() == 2:
                        t_source += [spec.size(1)]
                        tensors += [spec]
                    else:
                        t_source += [spec.size(2)]
                        tensors += spec
                    del spec

                    targets += [tg.unsqueeze(0)]
                    t_len += [len(tg)]

                    k = k + 1
                    del waveform
                    del label

                else:
                    print('REMOVED:', ut_id, ' LAB:', label)

            if tensors and targets:
                tensors = pad_sequence(tensors, 0)
                targets = pad_sequence(targets, PAD_token)
                o_batch = [tensors.squeeze(1), targets.squeeze(1),
                           torch.tensor(t_len), torch.tensor(t_source)]

                out_batch.append(o_batch)

        return out_batch
        # return c_tensors, c_targets, c_t_len, c_t_source


class CollateInferFn(object):
    def __init__(self, args):
        self.args = args
        self.glottal_feat_map = None
        self.glottal_dim = 0
        self._glottal_norm_mean = None
        self._glottal_norm_std = None
        self._missing_glottal_warned = False
        # glottal_missing_count: utterances that used a zero glottal tensor (CSV id not in map).
        # glottal_found_count: CSV hit, or every utterance when glottal_from_waveform.
        self.glottal_missing_count = 0
        self.glottal_found_count = 0

        if getattr(args, "append_glottal_features", False):
            if getattr(args, "glottal_from_waveform", False):
                from extract_glottal_features_qcp import QCP_FRAME_FEATURE_DIM

                self.glottal_dim = QCP_FRAME_FEATURE_DIM
                self.glottal_feat_map = None
                stats_in = getattr(args, "glottal_norm_stats_in", None)
                if getattr(args, "glottal_standardize", True) and stats_in:
                    self._glottal_norm_mean, self._glottal_norm_std = _load_standardization_stats(
                        stats_in, self.glottal_dim
                    )
            else:
                if not args.glottal_features_path:
                    raise ValueError(
                        "--glottal_features_path is required when --append_glottal_features is set "
                        "(unless --glottal_from_waveform is set)"
                    )
                self.glottal_feat_map, self.glottal_dim = load_glottal_feature_map(
                    args.glottal_features_path,
                    drop_mfcc=getattr(args, "glottal_drop_mfcc", False),
                    standardize=getattr(args, "glottal_standardize", True),
                    stats_in_path=getattr(args, "glottal_norm_stats_in", None),
                    stats_out_path=getattr(args, "glottal_norm_stats_out", None),
                    utt_level_mean=getattr(args, "glottal_utt_mean", False),
                )

    def __call__(self, batch,
                 SOS_token=None, EOS_token=None, PAD_token=None):
        if SOS_token == None:
            SOS_token = self.args.trg_sos_idx
        if EOS_token == None:
            EOS_token = self.args.trg_eos_idx
        if PAD_token == None:
            PAD_token = self.args.trg_pad_idx

        tensors, targets, t_source = [], [], []

        # Gather in lists, and encode labels as indices
        for sample in batch:
            waveform, smp_freq, label, spk_id = sample[:4]
            ut_id = _extract_utt_id(sample)
            smp_freq = int(smp_freq)
            label = re.sub(r"[#^$,?:;.!]+|<unk>", "", label)

            if "ignore_time_segment_in_scoring" in label:
                continue
            if getattr(self.args, "use_precomputed_features", False):
                spec = waveform.float()
                if spec.dim() == 1:
                    spec = spec.unsqueeze(1)
                if spec.dim() == 3:
                    spec = spec.squeeze(0)
                spec = _truncate_feature_time(spec, self.args, ut_id)
            else:
                spec = spec_transform(waveform, self.args)  # .to(self.args.device)
                spec = melspec_transform(spec, self.args)

                if spec.dim() == 3:
                    spec = spec.squeeze(0)
                spec = _truncate_feature_time(spec, self.args, ut_id)
            if spec.size(1) < _min_mel_frames(self.args):
                continue

            if getattr(self.args, "append_glottal_features", False) and not getattr(
                self.args, "use_precomputed_features", False
            ):
                if getattr(self.args, "glottal_from_waveform", False):
                    g_rep = compute_glottal_features_from_waveform(
                        waveform,
                        smp_freq,
                        spec.size(1),
                        self.args,
                        norm_mean=self._glottal_norm_mean,
                        norm_std=self._glottal_norm_std,
                    )
                    g_rep = g_rep.to(spec.device, dtype=spec.dtype)
                    self.glottal_found_count += 1
                else:
                    uid = _normalize_utt_id(ut_id)
                    g = self.glottal_feat_map.get(uid)
                    if g is None:
                        g_rep = torch.zeros(self.glottal_dim, spec.size(1), dtype=torch.float32)
                        self.glottal_missing_count += 1
                        if not self._missing_glottal_warned:
                            print(f"WARNING: missing glottal features for utterance '{uid}'. Using zeros.")
                            self._missing_glottal_warned = True
                    else:
                        self.glottal_found_count += 1
                        g = g.to(spec.device)
                        if g.dim() == 1:
                            g_rep = g.unsqueeze(1).repeat(1, spec.size(1))
                        elif g.dim() == 2:
                            if g.size(1) == spec.size(1):
                                g_rep = g
                            else:
                                g_rep = F.interpolate(
                                    g.unsqueeze(0),
                                    size=spec.size(1),
                                    mode="linear",
                                    align_corners=False,
                                ).squeeze(0)
                        else:
                            g = g.reshape(self.glottal_dim, -1)
                            g_rep = F.interpolate(
                                g.unsqueeze(0),
                                size=spec.size(1),
                                mode="linear",
                                align_corners=False,
                            ).squeeze(0)
                spec = torch.cat([spec, g_rep], dim=0)

            if self.args.bpe == True:
                bpe_label = normalize_label_for_bpe_args(self.args, label)
                if not bpe_label:
                    continue
                tg = torch.LongTensor(
                    [self.args.sp.bos_id()]
                    + self.args.sp.encode_as_ids(bpe_label)
                    + [self.args.sp.eos_id()]
                )
            else:
                tg = torch.LongTensor(
                    text_transform.text_to_int("^"+label.lower()+"$"))

            t_source += [spec.size(1)]
            tensors += [spec]
            targets += [tg.unsqueeze(0)]
            del spec
            del waveform
            del label

        if tensors and targets:
            tensors = pad_sequence(tensors, 0)
            targets = pad_sequence(targets, PAD_token)
            return tensors.squeeze(1), targets.squeeze(1), torch.tensor(t_source)

        return None
