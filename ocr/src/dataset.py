import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
import pathlib
import cv2

from ocr.src.tokenizer import Tokenizer


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
        csv_data = pd.read_csv(csv_path)
        csv_data['dataset_name'] = csv_path
        csv_data['filename'] = csv_data['filename'].apply(
            get_full_img_path, csv_path=csv_path)
        data.append(csv_data[['filename', 'dataset_name', 'text']])
    data = pd.concat(data, ignore_index=True)
    return data


class DatasetProb2SampleProb:
    """Convert dataset sampling prob to sampling prob for each sample in the
    datset.

    Args:
        dataset_names (list): A list of the dataset names.
        dataset_probs (list of float): A list of dataset sample probs
            corresponding to the datasets from dataset_names list.
    """

    def __init__(self, dataset_names, dataset_probs):
        assert len(dataset_names) == len(dataset_probs), "Length of " \
            "csv_paths should be equal to the length of the dataset_probs."
        self.dataset2dataset_prob = dict(zip(dataset_names, dataset_probs))

    def _get_dataset2sample_count(self, data):
        dataset2sample_count = {}
        for dataset_name in self.dataset2dataset_prob:
            dataset2sample_count[dataset_name] = \
                (data['dataset_name'] == dataset_name).sum()
        return dataset2sample_count

    def _get_dataset2sample_prob(self, dataset2sample_count):
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
        dataset2sample_count = self._get_dataset2sample_count(data)
        dataset2sample_prob = \
            self._get_dataset2sample_prob(dataset2sample_count)
        data['sample_prob'] = data['dataset_name'].apply(
            lambda x: dataset2sample_prob[x])
        return data


class DataPreprocess:
    """Data preprocessing: concatente datasets in one pandas DataFrame and
    encode texts to ints.

    Args:
        csv_paths (list): A list of the dataset csv paths.
        dataset_probs (list of float): A list of dataset sample probs
            corresponding to the datasets from csv_paths list.
        tokenizer (ocr.tokenizer.Tokenizer): Tokenizer class.

    Return:
        data (pandas.DataFrame): Preprocessed dataset.
    """

    def __init__(self, csv_paths, dataset_probs, tokenizer):
        self.csv_paths = csv_paths
        self.dataset_prob2sample_prob = DatasetProb2SampleProb(
            self.csv_paths, dataset_probs)
        self.tokenizer = tokenizer

    def __call__(self):
        data = read_and_concat_datasets(self.csv_paths)
        data['enc_text'] = self.tokenizer.encode(data['text'].values)
        data = self.dataset_prob2sample_prob(data)
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
