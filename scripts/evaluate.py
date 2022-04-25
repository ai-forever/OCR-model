import torch
import argparse

from ocr.dataset import get_data_loader
from ocr.utils import val_loop, configure_logging
from ocr.transforms import get_val_transforms
from ocr.tokenizer import Tokenizer, BeamSearcDecoder, BestPathDecoder
from ocr.config import Config
from ocr.models import CRNN


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    config = Config(args.config_path)
    tokenizer = Tokenizer(config.get('alphabet'))
    logger = configure_logging()

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

    if args.lm_path:
        decoder = BeamSearcDecoder(config.get('alphabet'), args.lm_path)
    else:
        decoder = BestPathDecoder(config.get('alphabet'))

    acc_avg = val_loop(test_loader, model, decoder, logger, DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='/workdir/scripts/ocr_config.json',
                        help='Path to config.json.')
    parser.add_argument('--model_path', type=str,
                        help='Path to model weights.')
    parser.add_argument('--lm_path', type=str, default='',
                        help='Path to KenLM language model .arpa.')
    args = parser.parse_args()

    main(args)
