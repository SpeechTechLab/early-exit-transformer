import re
import os
import csv
import torch
import torchaudio.transforms as T
import torch.nn.functional as F

# For loading precomputed features
import numpy as np


def _normalize_utt_id(raw_id):
    base = os.path.basename(str(raw_id))
    return os.path.splitext(base)[0]


def _select_numeric_feature_columns(fieldnames, drop_mfcc):
    meta_cols = {
        "file_name", "speaker", "label", "task", "utt_id", "ut_id", "id", "path",
        "chapter_id", "transcript", "text", "sentence"
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


def _standardize_features(feature_matrix, eps=1e-8):
    mean = np.mean(feature_matrix, axis=0)
    std = np.std(feature_matrix, axis=0)
    std = np.where(std < eps, 1.0, std)
    return (feature_matrix - mean) / std


def load_glottal_feature_map(csv_path, drop_mfcc=True, standardize=True):
    feat_dim = None
    per_utt_values = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Invalid CSV (no header): {csv_path}")

        cols = _select_numeric_feature_columns(reader.fieldnames, drop_mfcc)
        id_candidates = ["file_name", "utt_id", "ut_id", "id", "path"]

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

    feature_matrix = np.vstack(per_utt_matrix)
    if standardize:
        feature_matrix = _standardize_features(feature_matrix)

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


def infer_glottal_feature_dim(csv_path, drop_mfcc=True):
    _, feat_dim = load_glottal_feature_map(csv_path, drop_mfcc=drop_mfcc)
    return feat_dim


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
                        tg = torch.LongTensor(
                            [self.args.sp.bos_id()] + self.args.sp.encode_as_ids(label) + [self.args.sp.eos_id()])
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
        self._missing_glottal_warned = False

        if getattr(args, "append_glottal_features", False):
            if not args.glottal_features_path:
                raise ValueError("--glottal_features_path is required when --append_glottal_features is set")
            self.glottal_feat_map, self.glottal_dim = load_glottal_feature_map(
                args.glottal_features_path,
                drop_mfcc=getattr(args, "glottal_drop_mfcc", False),
                standardize=getattr(args, "glottal_standardize", True),
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

            for waveform, smp_freq, label, spk_id, ut_id, *_ in c_batch:
                label = re.sub(r"<unk>|\[ unclear \]", "", label)
                label = re.sub(r"[#^$?:;.!\[\]]+", "", label)

                if len(label) < self.args.max_utterance_length:
                    if getattr(self.args, "use_precomputed_features", False):
                        # Precomputed tensors are expected as [feature_dim, time].
                        spec = waveform.to(self.args.device).float()
                        if spec.dim() == 1:
                            spec = spec.unsqueeze(1)
                    else:
                        spec = spec_transform(waveform, self.args)  # .to(device)
                        spec = melspec_transform(
                            spec, self.args).to(self.args.device)

                    if spec.dim() == 3:
                        spec = spec.squeeze(0)

                    if getattr(self.args, "append_glottal_features", False):
                        uid = _normalize_utt_id(ut_id)
                        g = self.glottal_feat_map.get(uid)
                        if g is None:
                            g = torch.zeros(self.glottal_dim, dtype=torch.float32)
                            if not self._missing_glottal_warned:
                                print(f"WARNING: missing glottal features for utterance '{uid}'. Using zeros.")
                                self._missing_glottal_warned = True
                        g = g.to(spec.device)
                        g_rep = g.unsqueeze(1).repeat(1, spec.size(1))
                        spec = torch.cat([spec, g_rep], dim=0)

                    if spec.dim() == 2:
                        t_source += [spec.size(1)]
                        tensors += [spec]
                    else:
                        t_source += [spec.size(2)]
                        tensors += spec
                    del spec

                    if self.args.bpe == True:
                        tg = torch.LongTensor(
                            [self.args.sp.bos_id()] + self.args.sp.encode_as_ids(label) + [self.args.sp.eos_id()])
                    else:
                        tg = torch.LongTensor(
                            text_transform.text_to_int("^"+label.lower()+"$"))
                    targets += [tg.unsqueeze(0)]
                    t_len += [len(tg)]

                    k = k + 1
                    del waveform
                    del label

                else:
                    print('REMOVED:', ut_id, ' LAB:', label)

            if tensors:
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
        self._missing_glottal_warned = False

        if getattr(args, "append_glottal_features", False):
            if not args.glottal_features_path:
                raise ValueError("--glottal_features_path is required when --append_glottal_features is set")
            self.glottal_feat_map, self.glottal_dim = load_glottal_feature_map(
                args.glottal_features_path,
                drop_mfcc=getattr(args, "glottal_drop_mfcc", False),
                standardize=getattr(args, "glottal_standardize", True),
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
        for waveform, smp_freq, label, spk_id, ut_id, *_ in batch:
            label = re.sub(r"[#^$,?:;.!]+|<unk>", "", label)

            if "ignore_time_segment_in_scoring" in label:
                continue
            spec = spec_transform(waveform, self.args)  # .to(self.args.device)
            spec = melspec_transform(spec, self.args).to(self.args.device)

            if spec.dim() == 3:
                spec = spec.squeeze(0)

            if getattr(self.args, "append_glottal_features", False):
                uid = _normalize_utt_id(ut_id)
                g = self.glottal_feat_map.get(uid)
                if g is None:
                    g = torch.zeros(self.glottal_dim, dtype=torch.float32)
                    if not self._missing_glottal_warned:
                        print(f"WARNING: missing glottal features for utterance '{uid}'. Using zeros.")
                        self._missing_glottal_warned = True
                g = g.to(spec.device)
                g_rep = g.unsqueeze(1).repeat(1, spec.size(1))
                spec = torch.cat([spec, g_rep], dim=0)

            t_source += [spec.size(1)]

            tensors += [spec]
            del spec
            
            if self.args.bpe == True:
                tg = torch.LongTensor(
                    [self.args.sp.bos_id()] + self.args.sp.encode_as_ids(label) + [self.args.sp.eos_id()])
            else:
                tg = torch.LongTensor(
                    text_transform.text_to_int("^"+label.lower()+"$"))

            targets += [tg.unsqueeze(0)]
            del waveform
