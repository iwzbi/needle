from operator import index
import struct
import gzip
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return img[:, ::-1, :]
        else:
            return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        img_pad = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        x_start = self.padding + shift_x
        y_start = self.padding + shift_y
        return img_pad[x_start:x_start+img.shape[0], y_start:y_start+img.shape[1], :]


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            order = np.arange(len(self.dataset))
            np.random.shuffle(order)
            self.ordering = np.array_split(order, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        return self

    def __next__(self):
        if self.index >= len(self.ordering):
            raise StopIteration
        else:
            data_batch = self.dataset[self.ordering[self.index]]
            self.index += 1
        return tuple([Tensor(i) for i in data_batch])


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.images, self.labels = self._parse_mnist(image_filename, label_filename)
        super().__init__(transforms)

    def __getitem__(self, index) -> object:
        imgs = self.images[index].reshape((-1, 28, 28, 1))
        imgs = np.stack([self.apply_transforms(i) for i in imgs])
        imgs = imgs[0] if isinstance(index, int) else imgs
        return imgs, self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)
    def _parse_mnist(self, image_filesname, label_filename):
        """ Read an images and labels file in MNIST format.  See this page:
        http://yann.lecun.com/exdb/mnist/ for a description of the file format.
        Args:
            image_filename (str): name of gzipped images file in MNIST format
            label_filename (str): name of gzipped labels file in MNIST format
        Returns:
            Tuple (X,y):
                X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                    data.  The dimensionality of the data should be
                    (num_examples x input_dim) where 'input_dim' is the full
                    dimension of the data, e.g., since MNIST images are 28x28, it
                    will be 784.  Values should be of type np.float32, and the data
                    should be normalized to have a minimum value of 0.0 and a
                    maximum value of 1.0.
                y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                    labels of the examples.  Values should be of type np.int8 and
                    for MNIST will contain the values 0-9.
        """
        with gzip.open(label_filename, 'rb') as label_file:
            magic_num, num_label = struct.unpack('>II', label_file.read(8))
            y = np.frombuffer(label_file.read(), dtype=np.uint8)
        
        with gzip.open(image_filesname, 'rb') as image_file:
            magic_num, num_image, num_row, num_column = struct.unpack('>IIII', image_file.read(16))
            X = np.frombuffer(image_file.read(), dtype=np.uint8).reshape((len(y), 784)).astype(np.float32)
            X = (X - np.min(X)) / (np.max(X) - np.min(X))
        return X, y

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])

