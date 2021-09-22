import torch
import time
from tqdm import tqdm

from ocr.metrics import get_accuracy

from utils.utils import AverageMeter, sec2min


def val_loop(data_loader, model, tokenizer, device):
    acc_avg = AverageMeter()
    strat_time = time.time()
    model.eval()

    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, texts, _, _ in tqdm_data_loader:
        images = images.to(device)
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
