import torch
import numpy as np

from ocr.transforms import InferenceTransform
from ocr.tokenizer import Tokenizer
from ocr.config import Config
from ocr.models import CRNN


def predict(images, model, tokenizer, device):
    """Make model prediction.

    Args:
        images (torch.Tensor): Batch with tensor images.
        model (ocr.src.models.CRNN): OCR model.
        tokenizer (ocr.tokenizer.Tokenizer): Tokenizer class.
        device (torch.device): Torch device.
    """
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
    text_preds = tokenizer.decode(pred)
    return text_preds


class OcrPredictor:
    """Make OCR prediction.

    Args:
        model_path (str): The path to the model weights.
        config_path (str): The path to the model config.
        device (str): The device for computation. Default is cuda.
    """

    def __init__(self, model_path, config_path, device='cuda'):
        config = Config(config_path)
        self.tokenizer = Tokenizer(config.get('alphabet'))
        self.device = torch.device(device)
        # load model
        self.model = CRNN(
            number_class_symbols=self.tokenizer.get_num_chars(),
            pretrained=False
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

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

        images = self.transforms(images)
        pred = predict(images, self.model, self.tokenizer, self.device)

        if one_image:
            return pred[0]
        else:
            return pred
