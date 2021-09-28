from tqdm import tqdm
import os
import time
import torch
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from utils.utils import (
    load_pretrain_model, FilesLimitControl, AverageMeter, sec2min
)

from ocr.src.dataset import (
    OCRDataset, DataPreprocess, collate_fn, SequentialSampler, get_data_loader
)
from ocr.src.utils import val_loop
from ocr.src.transforms import get_train_transforms, get_val_transforms
from ocr.src.tokenizer import Tokenizer
from ocr.src.config import Config
from ocr.src.models import CRNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(data_loader, model, criterion, optimizer, epoch):
    loss_avg = AverageMeter()
    strat_time = time.time()
    model.train()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, texts, enc_pad_texts, text_lens in tqdm_data_loader:
        model.zero_grad()
        images = images.to(DEVICE)
        batch_size = len(texts)
        output = model(images)
        output_lenghts = torch.full(
            size=(output.size(1),),
            fill_value=output.size(0),
            dtype=torch.long
        )
        loss = criterion(output, enc_pad_texts, output_lenghts, text_lens)
        loss_avg.update(loss.item(), batch_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
    loop_time = sec2min(time.time() - strat_time)
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    print(f'\nEpoch {epoch}, Loss: {loss_avg.avg:.5f}, '
          f'LR: {lr:.7f}, loop_time: {loop_time}')
    return loss_avg.avg


def get_loaders(tokenizer, config):
    train_transforms = get_train_transforms(
        height=config.get_image('height'),
        width=config.get_image('width'),
        prob=0.2
    )
    train_loader = get_data_loader(
        transforms=train_transforms,
        csv_paths=config.get_train_datasets('csv_path'),
        tokenizer=tokenizer,
        dataset_probs=config.get_train_datasets('prob'),
        epoch_size=config.get_train('epoch_size'),
        batch_size=config.get_train('batch_size'),
        drop_last=True
    )
    val_transforms = get_val_transforms(
        height=config.get_image('height'),
        width=config.get_image('width')
    )
    val_loader = get_data_loader(
        transforms=val_transforms,
        csv_paths=config.get_val_datasets('csv_path'),
        tokenizer=tokenizer,
        dataset_probs=config.get_val_datasets('prob'),
        epoch_size=config.get_val('epoch_size'),
        batch_size=config.get_val('batch_size'),
        drop_last=False
    )
    return train_loader, val_loader


def main(args):
    config = Config(args.config_path)
    tokenizer = Tokenizer(config.get('alphabet'))
    os.makedirs(config.get('save_dir'), exist_ok=True)
    train_loader, val_loader = get_loaders(tokenizer, config)

    model = CRNN(number_class_symbols=tokenizer.get_num_chars())
    if config.get('pretrain_path'):
        states = load_pretrain_model(config.get('pretrain_path'), model)
        model.load_state_dict(states)
        print('Load pretrained model')
    model.to(DEVICE)

    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                  weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max',
                                  factor=0.5, patience=15)
    weight_limit_control = FilesLimitControl()
    best_acc = -np.inf

    acc_avg = val_loop(val_loader, model, tokenizer, DEVICE)
    for epoch in range(config.get('num_epochs')):
        loss_avg = train_loop(train_loader, model, criterion, optimizer, epoch)
        acc_avg = val_loop(val_loader, model, tokenizer, DEVICE)
        scheduler.step(acc_avg)
        if acc_avg > best_acc:
            best_acc = acc_avg
            model_save_path = os.path.join(
                config.get('save_dir'), f'model-{epoch}-{acc_avg:.4f}.ckpt')
            torch.save(model.state_dict(), model_save_path)
            print('Model weights saved')
            weight_limit_control(model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='/workdir/ocr/config.json',
                        help='Path to config.json.')
    args = parser.parse_args()

    main(args)
