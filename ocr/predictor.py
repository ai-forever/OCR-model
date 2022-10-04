import torch
import numpy as np

from ocr.transforms import InferenceTransform
from ocr.tokenizer import Tokenizer, BeamSearcDecoder, BestPathDecoder
from ocr.config import Config
from ocr.models import CRNN


def predict(images, model, decoder, device):
    """Make model prediction.

    Args:
        images (torch.Tensor): Batch with tensor images.
        model (ocr.src.models.CRNN): OCR model.
        decoder: BeamSearcDecoder or BestPathDecoder class from ocr.tokenizer.
        device (torch.device): Torch device.
    """
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    text_preds = decoder(output)
    model.train()
    return text_preds


def split_list2batches(lst, batch_size):
    """Split list of images to list of bacthes."""
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]


class OcrPredictor:
    """Make OCR prediction.

    Args:
        model_path (str): The path to the model weights.
        config_path (str): The path to the model config.
        device (str): The device for computation. Default is cuda.
    """

    def __init__(self, model_path, config_path, lm_path='', device='cuda'):
        config = Config(config_path)
        self.tokenizer = Tokenizer(config.get('alphabet'))
        self.device = torch.device(device)
        self.batch_size = config.get_test('batch_size')
        # load model
        self.model = CRNN(
            number_class_symbols=self.tokenizer.get_num_chars(),
            pretrained=False
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        if lm_path:
            self.decoder = BeamSearcDecoder(config.get('alphabet'), lm_path)
        else:
            self.decoder = BestPathDecoder(config.get('alphabet'))

        self.transforms = InferenceTransform(
            height=config.get_image('height'),
            width=config.get_image('width'),
        )

    def __call__(self, images):
        """
        Args:
            images (np.ndarray or list of np.ndarray): One image or list of
                images in BGR format.

        Returns:
            pred (str or list of strs): The predicted text for one input
                image, and a list with texts if there is a list of input images.
        """
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(f"Input must contain np.ndarray, "
                            f"tuple or list, found {type(images)}.")

        images_batches = split_list2batches(images, self.batch_size)
        pred = []
        for images_batch in images_batches:
            images_batch = self.transforms(images_batch)
            preds_batch = predict(
                images_batch, self.model, self.decoder, self.device)
            pred.extend(preds_batch)

        if one_image:
            return pred[0]
        else:
            return pred
