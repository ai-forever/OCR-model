import torch
import time
from tqdm import tqdm

from utils.utils import AverageMeter, sec2min

from ocr.src.metrics import get_accuracy
from ocr.src.predictor  import predict


def val_loop(data_loader, model, tokenizer, device):
    acc_avg = AverageMeter()
    strat_time = time.time()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, texts, _, _ in tqdm_data_loader:
        batch_size = len(texts)
        text_preds = predict(images, model, tokenizer, device)
        acc_avg.update(get_accuracy(texts, text_preds), batch_size)

    loop_time = sec2min(time.time() - strat_time)
    print(f'Validation, '
          f'acc: {acc_avg.avg:.4f}, loop_time: {loop_time}')
    return acc_avg.avg
