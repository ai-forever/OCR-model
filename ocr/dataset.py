import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
import pathlib
import cv2


class SequentialSampler(Sampler):
    """Make sequence of dataset indexes for batch sampler.
    Args:
        dataset_len (int): Length of train dataset.
        epoch_size (int, optional): Size of train epoch (by default it
            is equal to the dataset_len). Can be specified if you need to
            reduce the time of the epoch.
        init_sample_probs (list, optional): List of samples' probabilities to
            be added in batch. If None probs for all samples would be the same.
            The length of the list must be equal to the length of the dataset.
    """
    def __init__(self, dataset_len, epoch_size=None, init_sample_probs=None):
        self.dataset_len = dataset_len
        if epoch_size is not None:
            self.epoch_size = epoch_size
        else:
            self.epoch_size = dataset_len

        if init_sample_probs is None:
            self.init_sample_probs = \
                np.array([1. for i in range(dataset_len)], dtype=np.float64)
        else:
            self.init_sample_probs = \
                np.array(init_sample_probs, dtype=np.float64)
            assert len(self.init_sample_probs) == dataset_len, "The len " \
                "of the sample_probs must be equal to the dataset_len."
        self.init_sample_probs = \
            self._sample_probs_normalization(self.init_sample_probs)

    def _sample_probs_normalization(self, sample_probs):
        """Probabilities normalization to make them sum to 1.
        Sum might not be equal to 1 if probs are too small.
        """
        return sample_probs / sample_probs.sum()

    def __iter__(self):
        dataset_indexes = np.random.choice(
            a=self.dataset_len,
            size=self.epoch_size,
            p=self.init_sample_probs,
            replace=False,  # only unique samples inside an epoch
        )
        return iter(dataset_indexes)

    def __len__(self):
        return self.epoch_size


def collate_fn(batch):
    images, texts, enc_texts = zip(*batch)
    images = torch.stack(images, 0)
    text_lens = torch.LongTensor([len(text) for text in texts])
    enc_pad_texts = pad_sequence(enc_texts, batch_first=True, padding_value=0)
    return images, texts, enc_pad_texts, text_lens


def get_full_img_path(img_root_path, csv_path):
    """Merge csv root path and image name."""
    root_dir = pathlib.Path(csv_path).parent
    img_path = root_dir / pathlib.Path(img_root_path)
    return str(img_path)


def read_and_concat_datasets(csv_paths):
    """Read csv files and concatenate them into one pandas DataFrame.

    Args:
        csv_paths (list): List of the dataset csv paths.

    Return:
        data (pandas.DataFrame): Concatenated datasets.
    """
    data = []
    for csv_path in csv_paths:
        csv_data = pd.read_csv(
            csv_path, dtype={'text': 'str'}, keep_default_na=False)
        csv_data['dataset_name'] = csv_path
        csv_data['filename'] = csv_data['filename'].apply(
            get_full_img_path, csv_path=csv_path)
        data.append(csv_data[['filename', 'dataset_name', 'text']])
    data = pd.concat(data, ignore_index=True)
    return data


def get_data_loader(
    transforms, csv_paths, tokenizer, dataset_probs, epoch_size,
    batch_size, drop_last
):
    data = read_and_concat_datasets(csv_paths)
    data['enc_text'] = tokenizer.encode(data['text'].values)

    dataset_prob2sample_prob = DatasetProb2SampleProb(csv_paths, dataset_probs)
    data = dataset_prob2sample_prob(data)

    dataset = OCRDataset(data, transforms)
    sampler = SequentialSampler(
        dataset_len=len(data),
        epoch_size=epoch_size,
        init_sample_probs=data['sample_prob'].values
    )
    batcher = torch.utils.data.BatchSampler(sampler, batch_size=batch_size,
                                            drop_last=drop_last)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_sampler=batcher,
        num_workers=8,
    )
    return data_loader


class DatasetProb2SampleProb:
    """Convert dataset sampling probability to probability for each sample
    in the datset.

    Args:
        dataset_names (list): A list of the dataset names.
        dataset_probs (list of float): A list of dataset sample probs
            corresponding to the datasets from dataset_names list.
    """

    def __init__(self, dataset_names, dataset_probs):
        assert len(dataset_names) == len(dataset_probs), "Length of " \
            "csv_paths should be equal to the length of the dataset_probs."
        self.dataset2dataset_prob = dict(zip(dataset_names, dataset_probs))

    def _dataset2sample_count(self, data):
        """Calculate samples in each dataset from data using."""
        dataset2sample_count = {}
        for dataset_name in self.dataset2dataset_prob:
            dataset2sample_count[dataset_name] = \
                (data['dataset_name'] == dataset_name).sum()
        return dataset2sample_count

    def _dataset2sample_prob(self, dataset2sample_count):
        """Convert dataaset prob to sample prob."""
        dataset2sample_prob = {}
        for dataset_name, dataset_prob in self.dataset2dataset_prob.items():
            sample_count = dataset2sample_count[dataset_name]
            dataset2sample_prob[dataset_name] = dataset_prob / sample_count
        return dataset2sample_prob

    def __call__(self, data):
        """Add sampling prob column to data.

        Args:
            data (pandas.DataFrame): Dataset with 'dataset_name' column.
        """
        dataset2sample_count = self._dataset2sample_count(data)
        dataset2sample_prob = \
            self._dataset2sample_prob(dataset2sample_count)
        data['sample_prob'] = data['dataset_name'].apply(
            lambda x: dataset2sample_prob[x])
        return data


class OCRDataset(Dataset):
    """OCR torch.Dataset.

    Args:
        data (pandas.DataFrame): Dataset with 'filename', 'text' and
            'enc_text' columns.
        transform (torchvision.Compose): Image transforms, default is None.
    """

    def __init__(self, data, transform=None):
        super().__init__()
        self.transform = transform
        self.data_len = len(data)
        self.img_paths = data['filename'].values
        self.texts = data['text'].values
        self.enc_texts = data['enc_text'].values

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        text = self.texts[idx]
        enc_text = torch.LongTensor(self.enc_texts[idx])
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, text, enc_text
