from tqdm import tqdm
import os
import time
import torch
import argparse
import numpy as np

from ocr.utils import (
    val_loop, load_pretrain_model, FilesLimitControl, AverageMeter, sec2min,
    configure_logging
)

from ocr.dataset import get_data_loader
from ocr.transforms import get_train_transforms, get_val_transforms
from ocr.tokenizer import Tokenizer, BestPathDecoder
from ocr.config import Config
from ocr.models import CRNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(
    data_loader, model, criterion, optimizer, epoch, scheduler, logger
):
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
        scheduler.step()
    loop_time = sec2min(time.time() - strat_time)
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    logger.info(f'Epoch {epoch}, Loss: {loss_avg.avg:.5f}, '
                f'LR: {lr:.7f}, loop_time: {loop_time}')
    return loss_avg.avg


def get_loaders(tokenizer, config):
    train_transforms = get_train_transforms(
        height=config.get_image('height'),
        width=config.get_image('width'),
        prob=0.4
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
    log_path = os.path.join(config.get('save_dir'), "output.log")
    logger = configure_logging(log_path)
    train_loader, val_loader = get_loaders(tokenizer, config)

    model = CRNN(number_class_symbols=tokenizer.get_num_chars())
    if config.get('pretrain_path'):
        states = load_pretrain_model(
            config.get('pretrain_path'), model, logger)
        model.load_state_dict(states)
        logger.info(f"Load pretrained model {config.get('pretrain_path')}")
    model.to(DEVICE)

    decoder = BestPathDecoder(config.get('alphabet'))

    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        epochs=config.get('num_epochs'),
        steps_per_epoch=len(train_loader),
        max_lr=0.001,
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=10 ** 5
    )
    weight_limit_control = FilesLimitControl(logger=logger)
    best_acc = -np.inf

    acc_avg = val_loop(val_loader, model, decoder, logger, DEVICE)
    for epoch in range(config.get('num_epochs')):
        loss_avg = train_loop(train_loader, model, criterion, optimizer,
                              epoch, scheduler, logger)
        acc_avg = val_loop(val_loader, model, decoder, logger, DEVICE)
        if acc_avg > best_acc:
            best_acc = acc_avg
            model_save_path = os.path.join(
                config.get('save_dir'), f'model-{epoch}-{acc_avg:.4f}.ckpt')
            torch.save(model.state_dict(), model_save_path)
            logger.info(f'Model weights saved {model_save_path}')
            weight_limit_control(model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='/workdir/scripts/ocr_config.json',
                        help='Path to config.json.')
    args = parser.parse_args()

    main(args)
