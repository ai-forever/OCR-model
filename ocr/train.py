from tqdm import tqdm
import os
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from utils.utils import (
    load_pretrain_model, FilesLimitControl, AverageMeter, sec2min
)

from metrics import get_accuracy
from dataset import OCRDataset, DataPreprocess, collate_fn, SequentialSampler
from transforms import get_train_transforms, get_val_transforms
from tokenizer import Tokenizer
from config import CONFIG
from models import CRNN

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


def val_loop(data_loader, model, tokenizer):
    acc_avg = AverageMeter()
    strat_time = time.time()
    model.eval()

    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, texts, _, _ in tqdm_data_loader:
        images = images.to(DEVICE)
        batch_size = len(texts)
        with torch.no_grad():
            output = model(images)
        predicted_sequence = \
            torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
        text_preds = tokenizer.decode(predicted_sequence)
        acc_avg.update(get_accuracy(texts, text_preds), batch_size)

    loop_time = sec2min(time.time() - strat_time)
    print(f'Validation, '
          f'acc: {acc_avg.avg:.4f}, loop_time: {loop_time}')
    return acc_avg.avg


def get_loader(
    transforms, csv_paths, tokenizer, dataset_probs, epoch_size,
    batch_size, drop_last
):
    data_process = DataPreprocess(
        csv_paths=csv_paths,
        tokenizer=tokenizer,
        dataset_probs=dataset_probs
    )
    data = data_process()
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


def get_loaders(tokenizer):
    train_transforms = get_train_transforms(
        height=CONFIG.get_image('height'),
        width=CONFIG.get_image('width'),
        prob=0.2
    )
    train_loader = get_loader(
        transforms=train_transforms,
        csv_paths=CONFIG.get_train_datasets('csv_path'),
        tokenizer=tokenizer,
        dataset_probs=CONFIG.get_train_datasets('prob'),
        epoch_size=CONFIG.get_train('epoch_size'),
        batch_size=CONFIG.get_train('batch_size'),
        drop_last=True
    )
    val_transforms = get_val_transforms(
        height=CONFIG.get_image('height'),
        width=CONFIG.get_image('width')
    )
    val_loader = get_loader(
        transforms=val_transforms,
        csv_paths=CONFIG.get_val_datasets('csv_path'),
        tokenizer=tokenizer,
        dataset_probs=CONFIG.get_val_datasets('prob'),
        epoch_size=CONFIG.get_val('epoch_size'),
        batch_size=CONFIG.get_val('batch_size'),
        drop_last=False
    )
    return train_loader, val_loader


def main():
    os.makedirs(CONFIG.get('save_dir'), exist_ok=True)
    tokenizer = Tokenizer(CONFIG.get('alphabet'))
    train_loader, val_loader = get_loaders(tokenizer)

    model = CRNN(number_class_symbols=tokenizer.get_num_chars())
    if CONFIG.get('pretrain_path'):
        states = load_pretrain_model(CONFIG.get('pretrain_path'), model)
        model.load_state_dict(states)
        print('Load pretrained model')
    model.to(DEVICE)

    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                  weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                  factor=0.5, patience=50)
    weight_limit_control = FilesLimitControl()
    best_acc = -np.inf

    acc_avg = val_loop(val_loader, model, tokenizer)
    for epoch in range(CONFIG.get('num_epochs')):
        loss_avg = train_loop(train_loader, model, criterion, optimizer, epoch)
        acc_avg = val_loop(val_loader, model, tokenizer)
        scheduler.step(loss_avg)
        if acc_avg > best_acc:
            best_acc = acc_avg
            model_save_path = os.path.join(
                CONFIG.get('save_dir'), f'model-{epoch}-{acc_avg:.4f}.ckpt')
            torch.save(model.state_dict(), model_save_path)
            print('Model weights saved')
            weight_limit_control(model_save_path)


if __name__ == '__main__':
    main()
