import torch
import torchaudio

from util.data_loader import CollatePaddingFn, CollateInferFn


def get_data_loader(args):
    try:
        train_dataset1 = torchaudio.datasets.LIBRISPEECH(
            "", url="train-clean-100", download=False
        )
        train_dataset2 = torchaudio.datasets.LIBRISPEECH(
            "", url="train-clean-360", download=False
        )
        train_dataset3 = torchaudio.datasets.LIBRISPEECH(
            "", url="train-other-500", download=False
        )
    except RuntimeError as e:
        if "Dataset not found" not in str(e):
            raise
        print("LibriSpeech full training set not found locally. Downloading missing splits...")
        train_dataset1 = torchaudio.datasets.LIBRISPEECH(
            "", url="train-clean-100", download=True
        )
        train_dataset2 = torchaudio.datasets.LIBRISPEECH(
            "", url="train-clean-360", download=True
        )
        train_dataset3 = torchaudio.datasets.LIBRISPEECH(
            "", url="train-other-500", download=True
        )

    train_dataset = torch.utils.data.ConcatDataset(
        [train_dataset1, train_dataset2, train_dataset3]
    )

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

    try:
        try:
            train_dataset = torchaudio.datasets.LIBRISPEECH(
                "", url=split, download=False
            )
        except RuntimeError as e:
            if "Dataset not found" not in str(e):
                raise
            print(f"LibriSpeech {split} not found locally. Downloading dataset...")
            train_dataset = torchaudio.datasets.LIBRISPEECH(
                "", url=split, download=True
            )

        collate_infer_fn = CollateInferFn(args=args)
        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            pin_memory=False,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate_infer_fn,
            num_workers=args.n_workers,
        )
        return data_loader

    except Exception:
        exit("Invalid data split")