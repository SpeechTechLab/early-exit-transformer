import json
import os
import sys
import re
from torch import nn, optim
import torchaudio
from torchaudio.models.decoder import ctc_decoder

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


def evaluate_batch_ae(args, model, batch, valid_len, split, inf, vocab):
    beam_size = 10
    m = 5 / 200  # for deciding maximum length
    # p = 33 # for deciding maximum length for 5000
    p = 30  # for deciding maximum length for 256

    # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
    trg_expect = batch[1][:, 1:].to(args.device)

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

    for enc in encoder:
        i = i + 1
        batch_hyps = []

        best_combined = inf.ctc_cuda_predict(enc, args.tokens)

        for best_ in best_combined:
            if args.bpe == True:
                hyp = apply_lex(args.sp.decode(best_[0].tokens).lower(), vocab)
            else:
                hyp = apply_lex(re.sub(r"[#^$]+", "", best_.lower()), vocab)
            print(split, "BEAM_OUT_", i, ":", hyp)
            batch_hyps.append(hyp)

        if wer_stats is not None and batch_refs is not None:
            wer_stats[i]["refs"].extend(batch_refs)
            wer_stats[i]["hyps"].extend(batch_hyps)

    return


def run(args, model, data_loader, split, inf, vocab, wer_stats=None):
    for batch in data_loader:
        if batch is None:
            continue
        # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
        trg_expect = batch[1][:, 1:].to(args.device)
        # cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28]
        # trg = batch[1][:, :-1].to(args.device)

        batch_refs = []
        for trg_expect_ in trg_expect:
            if args.bpe == True:
                ref = args.sp.decode(trg_expect_.squeeze(0).tolist()).lower()
            else:
                ref = re.sub(r"[#^$]+", "", text_transform.int_to_text(trg_expect_.squeeze(0)))
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

    input_features_length = args.n_mels
    if args.use_precomputed_features:
        if args.n_glottal_features <= 0:
            raise ValueError(
                "When --use_precomputed_features is set, --n_glottal_features must be > 0"
            )
        input_features_length = args.n_glottal_features
    elif args.append_glottal_features:
        if not args.glottal_features_path:
            raise ValueError(
                "When --append_glottal_features is set, --glottal_features_path must be provided"
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

    for split in ["test-clean", "test-other"]:  # "dev-clean", "dev-other":
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
            missing = getattr(collate_fn, "glottal_missing_count", 0)
            total = found + missing
            if total > 0:
                cov = 100.0 * found / total
                print(f"[Glottal coverage: {found}/{total} ({cov:.2f}%)]")
            else:
                print("[Glottal coverage: no items counted]")

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
            split_results[f"exit_{i}"] = {"wer_pct": wer_val, "utterances": len(refs)}
    print(sep)
    print()
    results[split] = split_results


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
