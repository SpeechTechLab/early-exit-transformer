import json
import os
import sys
import re
from torch import nn, optim
import os

# Avoid TorchCodec backend (requires FFmpeg libs).
os.environ.setdefault("TORCHAUDIO_USE_TORCHCODEC", "0")

import torchaudio

try:
    torchaudio.set_audio_backend("sox_io")
except Exception:
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass

# Do not import torchaudio.models.decoder.ctc_decoder here: it requires flashlight-text
# and would crash before BeamInference can fall back to greedy CTC. Decoding uses
# util.beam_infer.BeamInference only.

try:
    import jiwer
except ImportError:
    jiwer = None

from data import get_infer_data_loader
from models.model.early_exit import Early_conformer, full_conformer, Early_zipformer, Early_zipformer_2layer_exits, Splitformer
from util.beam_infer import BeamInference
from util.conf import get_args
from util.data_loader import text_transform
from util.epoch_timer import epoch_time
from util.model_utils import *
from util.tokenizer import *
from util.data_loader import infer_glottal_feature_dim


def _flatten_token_ids(ids):
    """Turn label token ids into a flat list[int] (handles [L, 1] collate layouts)."""
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().tolist()
    flat = []
    for x in ids:
        if isinstance(x, list):
            flat.extend(int(v) for v in x)
        else:
            flat.append(int(x))
    return flat


def _decode_bpe_ids(args, token_ids, *, skip_special: bool) -> str:
    """Decode BPE token ids; flatten [L, 1] layouts from collate."""
    flat = _flatten_token_ids(token_ids)
    if skip_special:
        skip = {
            int(getattr(args, "trg_pad_idx", 126)),
            int(getattr(args, "trg_sos_idx", 1)),
            int(getattr(args, "trg_eos_idx", 2)),
        }
        if hasattr(args, "sp") and args.sp is not None:
            skip.add(int(args.sp.pad_id()))
        flat = [i for i in flat if i not in skip]
    if not flat:
        return ""
    return args.sp.decode(flat).lower()


def _decode_reference(args, token_ids) -> str:
    """Decode reference labels for WER / EXPECTED (skip pad/bos/eos)."""
    if not getattr(args, "bpe", True):
        return re.sub(r"[#^$]+", "", text_transform.int_to_text(token_ids))
    return _decode_bpe_ids(args, token_ids, skip_special=True)


def _label_batch_matrix(batch_labels):
    """Normalize collate label tensor to [batch, time] before [:, 1:] shift."""
    labels = batch_labels
    if labels.dim() == 3 and labels.size(1) == 1:
        labels = labels.squeeze(1)
    return labels[:, 1:]


def evaluate_batch_ae(args, model, batch, valid_len, split, inf, vocab):
    beam_size = int(getattr(args, "beam_size", 10))
    m = 5 / 200  # for deciding maximum length
    # p = 33 # for deciding maximum length for 5000
    p = 30  # for deciding maximum length for 256

    trg_expect = _label_batch_matrix(batch[1]).to(args.device)

    for spec_, v_l, trg_expect_ in zip(batch[0], valid_len, trg_expect):

        if args.bpe == True:
            print(split, "\nEXPECTED:", args.sp.decode(
                trg_expect_.squeeze(0).tolist()).lower())
        else:
            print(split, "\nEXPECTED:", re.sub(
                r"[#^$]+", "", text_transform.int_to_text(trg_expect_.squeeze(0))))

        if spec_.size(1) < 200:
            max_len = int(p - spec_.size(1) * m)
        else:
            max_len = int(spec_.size(1) / 12)  # for 256
        min_len = int(max_len * 0.6)
        # print("MAX-MIN:", max_len, min_len,
        #       spec_.size(1), trg_expect_.size())

        for n in range(1, args.n_enc_exits+1):
            encoder_output = model._encoder_(
                spec_.unsqueeze(0), v_l.unsqueeze(0), n).to(args.device)

            out, scores, best_combined = inf.beam_search(
                model, encoder_output=encoder_output, layer_n=n,
                max_length=max_len, beam_size=beam_size,
                return_best_beam=True)

            del encoder_output

            if args.bpe == True:
                print(split, " BEAM_OUT_", n, ":",  apply_lex(
                    args.sp.decode(best_combined).lower(), vocab))

        del spec_
        del v_l

    return


def evaluate_batch_ctc(args, model, batch, valid_len, split, inf, vocab,
                       batch_refs=None, wer_stats=None):
    encoder = model(batch[0].to(args.device), valid_len)
    i = 0

    def _sp_decode_tokens(obj) -> str:
        return _decode_bpe_ids(args, obj, skip_special=False)

    for enc in encoder:
        i = i + 1
        batch_hyps = []

        best_combined = inf.ctc_cuda_predict(enc, args.tokens)

        for best_ in best_combined:
            if args.bpe == True:
                hyp = apply_lex(_sp_decode_tokens(best_[0].tokens), vocab)
            else:
                hyp = apply_lex(re.sub(r"[#^$]+", "", best_.lower()), vocab)
            print(split, "BEAM_OUT_", i, ":", hyp)
            batch_hyps.append(hyp)

        if wer_stats is not None and batch_refs is not None:
            wer_stats[i]["refs"].extend(batch_refs)
            wer_stats[i]["hyps"].extend(batch_hyps)
            # Save a small sample of ref/hyp pairs for debugging (written later).
            pairs = wer_stats[i].setdefault("pairs", [])
            max_pairs = int(getattr(args, "save_decodes_n", 0) or 0)
            if max_pairs > 0 and len(pairs) < max_pairs:
                for r, h in zip(batch_refs, batch_hyps):
                    if len(pairs) >= max_pairs:
                        break
                    pairs.append((r, h))

    return


def run(args, model, data_loader, split, inf, vocab, wer_stats=None):
    for batch in data_loader:
        if batch is None:
            continue
        trg_expect = _label_batch_matrix(batch[1]).to(args.device)

        batch_refs = []
        for trg_expect_ in trg_expect:
            ref = _decode_reference(args, trg_expect_)
            print(split, "EXPECTED:", ref)
            batch_refs.append(ref)

        valid_len = batch[2]

        if args.decoder_mode == 'aed':
            evaluate_batch_ae(args, model, batch,
                              valid_len, split, inf, vocab)
        elif args.decoder_mode == 'ctc':
            evaluate_batch_ctc(args, model, batch,
                               valid_len, split, inf, vocab,
                               batch_refs=batch_refs, wer_stats=wer_stats)

    return


def main():
    #
    #   CONFIG
    #

    # Parse config from command line arguments
    args = get_args()

    if getattr(args, "use_precomputed_features", False) and getattr(
        args, "append_glottal_features", False
    ):
        raise ValueError(
            "Do not combine --use_precomputed_features with --append_glottal_features for inference."
        )

    # Optional decode-sample saving (JSONL)
    _save_path = getattr(args, "save_decodes_path", None)
    if _save_path:
        try:
            globals()["_SAVE_DECODE_FH"] = open(_save_path, "w", encoding="utf-8")
            globals()["_SAVE_DECODE_N"] = int(getattr(args, "save_decodes_n", 50) or 50)
            print(f"[Saving decode samples to {_save_path} (up to {globals()['_SAVE_DECODE_N']} per exit per split)]")
        except Exception as exc:
            print(f"[Warning] Could not open save_decodes_path={_save_path!r}: {exc}")
            globals()["_SAVE_DECODE_FH"] = None
            globals()["_SAVE_DECODE_N"] = 0

    input_features_length = args.n_mels
    if args.use_precomputed_features:
        if args.n_glottal_features <= 0:
            raise ValueError(
                "When --use_precomputed_features is set, --n_glottal_features must be > 0"
            )
        input_features_length = args.n_glottal_features
    elif args.append_glottal_features:
        if getattr(args, "glottal_from_waveform", False):
            from extract_glottal_features_qcp import QCP_FRAME_FEATURE_DIM

            glottal_dim = args.n_glottal_features if args.n_glottal_features > 0 else QCP_FRAME_FEATURE_DIM
            args.n_glottal_features = glottal_dim
        else:
            if not args.glottal_features_path:
                raise ValueError(
                    "When --append_glottal_features is set, --glottal_features_path must be provided "
                    "(unless --glottal_from_waveform is set)"
                )
            if args.n_glottal_features > 0:
                glottal_dim = args.n_glottal_features
            else:
                glottal_dim = infer_glottal_feature_dim(
                    args.glottal_features_path,
                    drop_mfcc=getattr(args, "glottal_drop_mfcc", False),
                )
                args.n_glottal_features = glottal_dim
        input_features_length = args.n_mels + glottal_dim

    #
    #   MODEL
    #

    # Define model
    if args.decoder_mode == 'aed':
        model = full_conformer(trg_pad_idx=args.trg_pad_idx,
                               n_enc_exits=args.n_enc_exits,
                               d_model=args.d_model,
                               enc_voc_size=args.enc_voc_size,
                               dec_voc_size=args.dec_voc_size,
                               max_len=args.max_len,
                               d_feed_forward=args.d_feed_forward,
                               n_head=args.n_heads,
                               n_enc_layers=args.n_enc_layers_per_exit,
                               n_dec_layers=args.n_dec_layers,
                               features_length=input_features_length,
                               drop_prob=args.drop_prob,
                               depthwise_kernel_size=args.depthwise_kernel_size,
                               device=args.device).to(args.device)

    elif args.decoder_mode == 'ctc':
        if args.model_type == 'early_conformer':
            model = Early_conformer(src_pad_idx=args.src_pad_idx,
                                    n_enc_exits=args.n_enc_exits,
                                    d_model=args.d_model,
                                    enc_voc_size=args.enc_voc_size,
                                    dec_voc_size=args.dec_voc_size,
                                    max_len=args.max_len,
                                    d_feed_forward=args.d_feed_forward,
                                    n_head=args.n_heads,
                                    n_enc_layers=args.n_enc_layers_per_exit,
                                    features_length=input_features_length,
                                    drop_prob=args.drop_prob,
                                    depthwise_kernel_size=args.depthwise_kernel_size,
                                    device=args.device).to(args.device)

        elif args.model_type == 'early_zipformer':
            model = Early_zipformer(src_pad_idx=args.src_pad_idx,
                                    n_enc_exits=args.n_enc_exits,
                                    d_model=args.d_model,
                                    enc_voc_size=args.enc_voc_size,
                                    dec_voc_size=args.dec_voc_size,
                                    max_len=args.max_len,
                                    d_feed_forward=args.d_feed_forward,
                                    n_head=args.n_heads,
                                    n_enc_layers=args.n_enc_layers_per_exit,
                                    features_length=input_features_length,
                                    drop_prob=args.drop_prob,
                                    depthwise_kernel_size=args.depthwise_kernel_size,
                                    device=args.device).to(args.device)

        elif args.model_type == 'early_zipformer_2layer_exits':
            model = Early_zipformer_2layer_exits(src_pad_idx=args.src_pad_idx,
                                    n_enc_exits=args.n_enc_exits,
                                    d_model=args.d_model,
                                    enc_voc_size=args.enc_voc_size,
                                    dec_voc_size=args.dec_voc_size,
                                    max_len=args.max_len,
                                    d_feed_forward=args.d_feed_forward,
                                    n_head=args.n_heads,
                                    n_enc_layers=args.n_enc_layers_per_exit,
                                    features_length=input_features_length,
                                    drop_prob=args.drop_prob,
                                    depthwise_kernel_size=args.depthwise_kernel_size,
                                    device=args.device).to(args.device)

        elif args.model_type == 'splitformer':
            model = Splitformer(src_pad_idx=args.src_pad_idx,
                                    n_enc_exits=args.n_enc_exits,
                                    d_model=args.d_model,
                                    enc_voc_size=args.enc_voc_size,
                                    dec_voc_size=args.dec_voc_size,
                                    max_len=args.max_len,
                                    d_feed_forward=args.d_feed_forward,
                                    n_head=args.n_heads,
                                    n_enc_layers=args.n_enc_layers_per_exit,
                                    features_length=input_features_length,
                                    drop_prob=args.drop_prob,
                                    depthwise_kernel_size=args.depthwise_kernel_size,
                                    device=args.device).to(args.device)

    else:
        raise ValueError(
            "Invalid decoder mode. Use either \"aed\" or \"ctc\" with --decoder_mode.")

    # If model checkpoint path is provided, load it.
    # (Overrides --load_model-dir)
    if args.load_model_path != None:
        path = os.getcwd() + '/' + args.load_model_path
        model.load_state_dict(torch.load(
            path, map_location=args.device))

    # If model checkpoint dir is provided, check that
    # the epochs to begin and end averaging are also
    # provided. If so, average the specified models.
    elif None not in (args.load_model_dir, args.avg_model_start, args.avg_model_end):
        model = avg_models(args, model, args.load_model_dir,
                           args.avg_model_start, args.avg_model_end)

    # If neither option has been provided, then raise error.
    else:
        raise ValueError(
            "Invalid model loading config. Use either --load_model_path for a single model or --load_model_dir/--avg_model_start/--avg_model_end for an average of models.")

    model.eval()

    print(f'The model has {count_parameters(model):,} trainable parameters')
    # print("batch_size:", batch_size, " num_heads:", n_heads, " num_encoder_layers:",
    #     n_enc_layers, "vocab_size:", dec_voc_size, "DEVICE:", device)

    torch.multiprocessing.set_start_method('spawn')
    torch.set_num_threads(args.n_threads)

    # Used to access various inference functions, see util/beam_infer
    inf = BeamInference(args=args)

    file_dict = 'librispeech.lex'
    vocab = load_dict(file_dict)

    results = {}

    if getattr(args, "use_precomputed_features", False):
        if not getattr(args, "manifest", None):
            raise ValueError(
                "Inference with --use_precomputed_features requires --manifest (eval utterances: "
                "csv_path,transcript per line)."
            )
        from pathlib import Path

        split = Path(args.manifest).stem
        infer_splits = [split]
        print(
            f"INFO: --use_precomputed_features: using --manifest={args.manifest!r} only; "
            f"--infer_splits is ignored (WER table tag: {split})."
        )
    else:
        infer_splits = [s.strip() for s in str(getattr(args, "infer_splits", "") or "").split(",") if s.strip()]
        if not infer_splits:
            infer_splits = ["test-clean", "test-other"]

    for split in infer_splits:  # e.g. "test-clean", "test-other", "train-clean-100"
        print(split)

        wer_stats = {i: {"refs": [], "hyps": []} for i in range(1, args.n_enc_exits + 1)}

        # Load data split
        data_loader = get_infer_data_loader(
            args=args, split=split, shuffle=False)


        with torch.no_grad():
            run(model=model, args=args, data_loader=data_loader,
                split=split, inf=inf, vocab=vocab, wer_stats=wer_stats)

        if getattr(args, "append_glottal_features", False) and hasattr(data_loader, "collate_fn"):
            collate_fn = data_loader.collate_fn
            found = getattr(collate_fn, "glottal_found_count", 0)
            zero_fill = getattr(collate_fn, "glottal_missing_count", 0)
            total = found + zero_fill
            if total > 0:
                if getattr(args, "glottal_from_waveform", False):
                    print(
                        f"[Infer glottal ({split}): {found} utterances with QCP-from-waveform; "
                        f"zero-filled N/A (glottal_from_waveform)]"
                    )
                else:
                    zpct = 100.0 * zero_fill / total
                    print(
                        f"[Infer glottal ({split}): {found} utterances used CSV glottal; "
                        f"{zero_fill} utterances had all-zero glottal (id missing from CSV; "
                        f"{zpct:.2f}% of {total})]"
                    )
            else:
                if getattr(args, "n_workers", 0) > 0:
                    print(
                        "[Infer glottal: counts unavailable with n_workers>0; "
                        "rerun with --n_workers 0 for zero-fill vs CSV stats]"
                    )
                else:
                    print("[Infer glottal: no items counted on collate_fn]")

        _print_wer_table(split, wer_stats, args.n_enc_exits, results)

    if results and jiwer is not None:
        _write_results(results, args)


def _print_wer_table(split, wer_stats, n_enc_exits, results):
    if jiwer is None:
        print("\n[jiwer not installed — skipping WER. Run: pip install jiwer]\n")
        return

    col_w = 12
    header = f"{'Exit':<6}" + f"{'WER (%)':>{col_w}}" + f"{'Utterances':>{col_w}}"
    sep = "-" * len(header)
    print(f"\n{'=== WER: ' + split + ' ===':^{len(header)}}")
    print(header)
    print(sep)

    split_results = {}
    for i in range(1, n_enc_exits + 1):
        refs = wer_stats[i]["refs"]
        hyps = wer_stats[i]["hyps"]
        if refs:
            wer_val = round(jiwer.wer(refs, hyps) * 100, 2)
            print(f"{i:<6}{wer_val:>{col_w}.2f}{len(refs):>{col_w}}")

            # INS/DEL/SUB breakdown (word-level) when available.
            ins = dele = sub = None
            try:
                if hasattr(jiwer, "compute_measures"):
                    m = jiwer.compute_measures(refs, hyps)
                    ins = int(m.get("insertions", 0))
                    dele = int(m.get("deletions", 0))
                    sub = int(m.get("substitutions", 0))
                elif hasattr(jiwer, "process_words"):
                    out = jiwer.process_words(refs, hyps)
                    ins = int(getattr(out, "insertions", 0))
                    dele = int(getattr(out, "deletions", 0))
                    sub = int(getattr(out, "substitutions", 0))
            except Exception:
                ins = dele = sub = None

            if None not in (ins, dele, sub):
                print(f"{'':<6}{'I/D/S':>{col_w}}{ins}/{dele}/{sub:>{col_w-4}}")

            split_results[f"exit_{i}"] = {
                "wer_pct": wer_val,
                "utterances": len(refs),
                "insertions": ins,
                "deletions": dele,
                "substitutions": sub,
            }
    print(sep)
    print()
    results[split] = split_results

    # Optional: save debug ref/hyp samples to JSONL
    save_path = getattr(sys.modules[__name__], "_SAVE_DECODE_FH", None)
    if save_path is not None:
        try:
            import json as _json
            for i in range(1, n_enc_exits + 1):
                for ref, hyp in wer_stats[i].get("pairs", []) or []:
                    save_path.write(_json.dumps({"split": split, "exit": i, "ref": ref, "hyp": hyp}, ensure_ascii=False) + "\n")
        except Exception:
            pass


def _write_results(results, args):
    if args.results_file:
        out_path = args.results_file
    else:
        # derive a default name from the model path used
        model_tag = (
            os.path.basename(args.load_model_path)
            if args.load_model_path
            else f"{args.load_model_dir}_avg{args.avg_model_start}-{args.avg_model_end}"
        )
        out_path = f"wer_{model_tag}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[WER results written to {out_path}]\n")


if __name__ == '__main__':
    main()
