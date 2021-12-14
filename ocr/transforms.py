import torch
import torchvision
import cv2
import random
import numpy as np


class RescalePaddingImage:
    def __init__(self, output_height, output_width):
        self.output_height = output_height
        self.output_width = output_width

    def __call__(self, image):
        h, w = image.shape[:2]
        # width proportional to change in  height
        new_width = int(w*(self.output_height/h))
        # new_width cannot be bigger than output_width
        new_width = min(new_width, self.output_width)
        image = cv2.resize(image, (new_width, self.output_height),
                           interpolation=cv2.INTER_LINEAR)
        if new_width < self.output_width:
            image = np.pad(
                image, ((0, 0), (0, self.output_width - new_width), (0, 0)),
                'constant', constant_values=0)
        return image


class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


class ToTensor:
    def __call__(self, arr):
        arr = torch.from_numpy(arr)
        return arr


class MoveChannels:
    """Move the channel axis to the zero position as required in pytorch."""

    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        else:
            return np.moveaxis(image, 0, -1)


class UseWithProb:
    def __init__(self, transform, prob=0.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = self.transform(image)
        return image


class RandomGaussianBlur:
    """Apply Gaussian blur with random kernel size

    Args:
        max_ksize (int): maximal size of a kernel to apply, should be odd
        sigma_x (int): Standard deviation
    """

    def __init__(self, max_ksize=5, sigma_x=20):
        assert max_ksize % 2 == 1, "max_ksize should be odd"
        self.max_ksize = max_ksize // 2 + 1
        self.sigma_x = sigma_x

    def __call__(self, image):
        kernal_size = tuple(2 * np.random.randint(0, self.max_ksize, 2) + 1)
        blured_image = cv2.GaussianBlur(image, kernal_size, self.sigma_x)
        return blured_image


class InferenceTransform:
    def __init__(self, height, width):
        self.transforms = get_val_transforms(height, width)

    def __call__(self, images):
        transformed_images = []
        for image in images:
            image = self.transforms(image)
            transformed_images.append(image)
        transformed_tensor = torch.stack(transformed_images, 0)
        return transformed_tensor


def get_train_transforms(height, width, prob):
    transforms = torchvision.transforms.Compose([
        RescalePaddingImage(height, width),
        UseWithProb(RandomGaussianBlur(max_ksize=3), prob=prob),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])
    return transforms


def get_val_transforms(height, width):
    transforms = torchvision.transforms.Compose([
        RescalePaddingImage(height, width),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])
    return transforms
