train_manifest=bridge2ai_zipformer_full_all_tasks/manifests/train.txt (30976 lines)
dev_manifest=bridge2ai_zipformer_full_all_tasks/manifests/dev.txt (3846 lines)
test_manifest=bridge2ai_zipformer_full_all_tasks/manifests/test.txt (6100 lines)
feature_dim=80 (mel only, no glottal)
seed=42

Train (do NOT pass --append_glottal_features):
  python3 train.py --use_precomputed_features \
    --manifest bridge2ai_zipformer_full_all_tasks/manifests/train.txt \
    --n_glottal_features 80 \
    --decoder_mode ctc --model_type early_zipformer_2layer_exits \
    --n_enc_layers_per_exit 2 --n_enc_exits 6 --n_epochs 30 \
    --batch_size 4 --n_workers 0 \
    --save_model_dir bridge2ai_zipformer_full_all_tasks/trained_model

Inference (held-out test manifest):
  python3 inference.py --use_precomputed_features \
    --manifest bridge2ai_zipformer_full_all_tasks/manifests/test.txt \
    --n_glottal_features 80 \
    --decoder_mode ctc --model_type early_zipformer_2layer_exits \
    --n_enc_layers_per_exit 2 --n_enc_exits 6 \
    --load_model_path bridge2ai_zipformer_full_all_tasks/trained_model/mod29-transformer \
    --results_file bridge2ai_zipformer_full_all_tasks/wer_test.json