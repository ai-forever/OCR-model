import torch
import onnxruntime as ort

from ocr.transforms import InferenceTransform
from ocr.tokenizer import Tokenizer, BeamSearcDecoder, BestPathDecoder
from ocr.config import Config
from ocr.models import CRNN


def predict(images, model, decoder, device):
    """Make model prediction.

    Args:
        images (torch.Tensor): Batch with tensor images.
        model (ocr.src.models.CRNN): OCR model.
        decoder: (ocr.tokenizer.OCRDecoder)
        device (torch.device): Torch device.
    """
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    text_preds = decoder.decode(output)
    return text_preds


def split_list2batches(lst, batch_size):
    """Split list of images to list of bacthes."""
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]


class OCRModel:
    def predict(self):
        raise NotImplementedError


class OCRONNXCPUModel(OCRModel):
    def __init__(self, model_path, config, decoder):
        self.tokenizer = Tokenizer(config.get('alphabet'))
        self.decoder = decoder
        # load model
        self.model = ort.InferenceSession(model_path)

        self.transforms = InferenceTransform(
            height=config.get_image('height'),
            width=config.get_image('width'),
            return_numpy=True
        )

    def predict(self, images):
        transformed_images = self.transforms(images)
        output = self.model.run(
            None,
            {"input": transformed_images},
        )[0]
        pred = self.decoder.decode_numpy(output)
        return pred


class OCRTorchModel(OCRModel):
    def __init__(self, model_path, config, decoder, device='cuda'):
        self.tokenizer = Tokenizer(config.get('alphabet'))
        self.device = torch.device(device)
        self.decoder = decoder
        # load model
        self.model = CRNN(
            number_class_symbols=self.tokenizer.get_num_chars(),
            pretrained=False
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=config.get_image('height'),
            width=config.get_image('width'),
        )

    def predict(self, images):
        transformed_images = self.transforms(images)
        pred = predict(
            transformed_images, self.model, self.decoder, self.device)
        return pred


class OcrPredictor:
    """Make OCR prediction.

    Args:
        model_path (str): The path to the model weights.
        config_path (str): The path to the model config.
        device (str): The device for computation. Default is cuda.
    """

    def __init__(
        self, model_path, config_path, lm_path='', device='cuda',
        batch_size=1, onnx=False
    ):
        self.batch_size = batch_size
        config = Config(config_path)
        if lm_path:
            decoder = BeamSearcDecoder(config.get('alphabet'), lm_path)
        else:
            decoder = BestPathDecoder(config.get('alphabet'))

        if onnx and device == 'cpu':
            self.model = OCRONNXCPUModel(model_path, config, decoder)
        elif onnx and device == 'cuda':
            raise Exception("ONNX runtime is only available on CPU devices")
        else:
            self.model = OCRTorchModel(model_path, config, decoder, device)

    def __call__(self, images):
        """
        Args:
            images (list of np.ndarray): A list of images in BGR format.

        Returns:
            pred (str or list of strs): The predicted text for one input
                image, and a list with texts if there was a list of images.
        """
        images_batches = split_list2batches(images, self.batch_size)
        pred = []
        for images_batch in images_batches:
            preds_batch = self.model.predict(images_batch)
            pred.extend(preds_batch)

        return pred
