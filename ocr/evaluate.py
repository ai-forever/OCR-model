import torch
import argparse

from utils.utils import load_pretrain_model

from ocr.src.dataset import (
    OCRDataset, DataPreprocess, collate_fn, SequentialSampler, get_data_loader
)
from ocr.src.utils import val_loop
from ocr.src.transforms import get_val_transforms
from ocr.src.tokenizer import Tokenizer
from ocr.src.config import Config
from ocr.src.models import CRNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    config = Config(args.config_path)
    tokenizer = Tokenizer(config.get('alphabet'))

    val_transforms = get_val_transforms(
        height=config.get_image('height'),
        width=config.get_image('width')
    )
    test_loader = get_data_loader(
        transforms=val_transforms,
        csv_paths=config.get_test_datasets('csv_path'),
        tokenizer=tokenizer,
        dataset_probs=config.get_test_datasets('prob'),
        epoch_size=config.get_test('epoch_size'),
        batch_size=config.get_test('batch_size'),
        drop_last=False
    )

    model = CRNN(number_class_symbols=tokenizer.get_num_chars())
    model.load_state_dict(torch.load(args.model_path))
    model.to(DEVICE)

    acc_avg = val_loop(test_loader, model, tokenizer, DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='/workdir/ocr/config.json',
                        help='Path to config.json.')
    parser.add_argument('--model_path', type=str,
                        help='Path to model weights.')
    args = parser.parse_args()

    main(args)
