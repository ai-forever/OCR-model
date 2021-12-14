import torch
import os
import math
import time
from tqdm import tqdm

from ocr.metrics import get_accuracy, wer, cer
from ocr.predictor import predict


def val_loop(data_loader, model, tokenizer, device):
    acc_avg = AverageMeter()
    wer_avg = AverageMeter()
    cer_avg = AverageMeter()
    strat_time = time.time()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, texts, _, _ in tqdm_data_loader:
        batch_size = len(texts)
        text_preds = predict(images, model, tokenizer, device)
        acc_avg.update(get_accuracy(texts, text_preds), batch_size)
        wer_avg.update(wer(texts, text_preds), batch_size)
        cer_avg.update(cer(texts, text_preds), batch_size)

    loop_time = sec2min(time.time() - strat_time)
    print(f'Validation, '
          f'acc: {acc_avg.avg:.4f}, '
          f'wer: {wer_avg.avg:.4f}, '
          f'cer: {cer_avg.avg:.4f}, '
          f'loop_time: {loop_time}')
    return acc_avg.avg


def sec2min(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FilesLimitControl:
    """Delete files from the disk if there are more files than the set limit.
    Args:
        max_weights_to_save (int, optional): The number of files that will be
            stored on the disk at the same time. Default is 3.
    """
    def __init__(self, max_weights_to_save=2):
        self.saved_weights_paths = []
        self.max_weights_to_save = max_weights_to_save

    def __call__(self, save_path):
        self.saved_weights_paths.append(save_path)
        if len(self.saved_weights_paths) > self.max_weights_to_save:
            old_weights_path = self.saved_weights_paths.pop(0)
            if os.path.exists(old_weights_path):
                os.remove(old_weights_path)
                print(f"Weigths removed '{old_weights_path}'")


def load_pretrain_model(weights_path, model):
    """Load the entire pretrain model or as many layers as possible.
    """
    old_dict = torch.load(weights_path)
    new_dict = model.state_dict()
    for key, weights in new_dict.items():
        if key in old_dict:
            if new_dict[key].shape == old_dict[key].shape:
                new_dict[key] = old_dict[key]
            else:
                print('Weights {} were not loaded'.format(key))
        else:
            print('Weights {} were not loaded'.format(key))
    return new_dict
