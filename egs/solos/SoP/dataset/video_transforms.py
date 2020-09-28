import random
import numbers
import torchvision.transforms.functional as F
from PIL import Image
import torch


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, frames):
        """
        Args:
            frames: a list of PIL Image
        Returns:
            a list of PIL Image: Rescaled images.
        """
        out_frames = []
        for frame in frames:
            out_frames.append(F.resize(frame, self.size, self.interpolation))
        return out_frames


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, frames):
        """
        Args:
            frames: a list of PIL Image
        Returns:
            a list of PIL Image: Cropped images.
        """
        out_frames = []
        for frame in frames:
            out_frames.append(F.center_crop(frame, self.size))
        return out_frames


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(frames, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            frames: a list of PIL Image
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = frames[0].size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, frames):
        """
        Args:
            frames: a list of PIL Image
        Returns:
            a list of PIL Image: Cropped images.
        """

        i, j, h, w = self.get_params(frames, self.size)

        out_frames = []
        for frame in frames:
            if self.padding is not None:
                frame = F.pad(frame, self.padding, self.fill, self.padding_mode)

            # pad the width if needed
            if self.pad_if_needed and frame.size[0] < self.size[1]:
                frame = F.pad(frame, (int((1 + self.size[1] - frame.size[0]) / 2), 0), self.fill, self.padding_mode)
            # pad the height if needed
            if self.pad_if_needed and frame.size[1] < self.size[0]:
                frame = F.pad(frame, (0, int((1 + self.size[0] - frame.size[1]) / 2)), self.fill, self.padding_mode)

            out_frames.append(F.crop(frame, i, j, h, w))
        return out_frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, frames):
        """
        Args:
            frames: a list of PIL Image
        Returns:
            a list of PIL Image: Flipped images.
        """

        if random.random() < self.p:
            out_frames = []
            for frame in frames:
                out_frames.append(F.hflip(frame))
            return out_frames
        else:
            return frames

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ToTensor(object):
    """Convert a list of ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a list of PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x L xH x W) in the range
    [0.0, 1.0].
    """

    def __call__(self, frames):
        """
        Args:
            frames: a list of (PIL Image or numpy.ndarray).
        Returns:
            a list of Tensor: Converted images.
        """
        out_frames = []
        for frame in frames:
            out_frames.append(F.to_tensor(frame))
        return out_frames


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            frames: a list of Tensor image of size (C, H, W) to be normalized.
        Returns:
            a list of Tensor: a list of normalized Tensor images.
        """
        out_frames = []
        for frame in frames:
            out_frames.append(F.normalize(frame, self.mean, self.std))
        return out_frames


class Stack(object):
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, frames):
        """
        Args:
            frames: a list of (L) Tensor image of size (C, H, W).
        Returns:
            Tensor: a video Tensor of size (C, L, H, W).
        """
        return torch.stack(frames, dim=self.dim)
