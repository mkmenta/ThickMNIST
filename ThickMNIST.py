import os
import struct
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.morphology import dilation, disk

__all__ = ["ThickMNIST"]


class ThickMNIST(Dataset):
    def __init__(self, split, path, transform_list=[], dilation_ratio=1, out_channels=1):
        assert dilation_ratio >= 0, "Dilation rate must be greater or equal to 0 (dilation_rate=0 means no dilation)"

        if split == 'train':
            fimages = os.path.join(path, 'raw', 'train-images-idx3-ubyte')
            flabels = os.path.join(path, 'raw', 'train-labels-idx1-ubyte')
        else:
            fimages = os.path.join(path, 'raw', 't10k-images-idx3-ubyte')
            flabels = os.path.join(path, 'raw', 't10k-labels-idx1-ubyte')

        # Load images
        with open(fimages, 'rb') as f:
            _, _, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.fromfile(f, dtype=np.uint8).reshape(-1, rows, cols)

        # Load labels
        with open(flabels, 'rb') as f:
            struct.unpack(">II", f.read(8))
            self.labels = np.fromfile(f, dtype=np.int8)
            self.labels = torch.from_numpy(self.labels.astype(np.int))

        self.transform_list = transform_list
        self.structuring_element = disk(dilation_ratio)
        if out_channels > 1:
            self.images = np.tile(self.images[:, :, :, np.newaxis], out_channels)
            self.structuring_element = np.tile(self.structuring_element[:, :, np.newaxis], out_channels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Range [0,255]
        label = self.labels[idx]

        image = dilation(image, self.structuring_element)

        image = Image.fromarray(image)
        for t in self.transform_list:
            image = t(image)
        image = transforms.ToTensor()(image)  # Range [0,1]

        return image, label
