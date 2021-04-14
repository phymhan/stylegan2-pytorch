from io import BytesIO

import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as F
import pickle
import os
import numpy as np
import tqdm
import random
from natsort import natsorted

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        # PIL size is (width, height), torchvision crop is (height, width)!!!
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0]
            else np.random.randint(low=0,high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
            else np.random.randint(low=0,high=img.size[1] - size[1]))
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class VideoFolderDataset(Dataset):
    def __init__(
        self,
        dataroot,
        transform=None,
        mode='video',
        min_len=8,
        frame_num=8,
        frame_step=1,
        nframe_num=2,  # Number of consecutive frames, when `mode`=='nframe'
        cache=None,
        unbind=True,
    ):
        assert(mode in ['video', 'image', 'nframe'])
        self.mode = mode
        self.root = dataroot
        self.cache = cache
        self.transform = transform or transforms.ToTensor()
        self.min_len = min_len
        self.frame_num = frame_num
        self.frame_step = frame_step
        self.nframe_num = nframe_num
        self.videos = []
        self.lengths = []
        self.unbind = unbind
        
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                cache_data = pickle.load(f)
            assert(isinstance(cache_data, dict))
            if not os.path.exists(dataroot) and 'root' in cache_data:
                self.root = cache_data['root']
            self.videos, self.lengths = cache_data['videos'], cache_data['lengths']
        else:
            video_list = []
            length_list = []
            for i, video in enumerate(tqdm.tqdm(os.listdir(dataroot), desc="Counting videos")):
                if os.path.isdir(os.path.join(dataroot, video)):
                    frames = natsorted(os.listdir(os.path.join(dataroot, video)))
                else:
                    continue
                frame_list = []
                for j, frame_name in enumerate(frames):
                    if is_image_file(os.path.join(dataroot, video, frame_name)):
                        # do not include dataroot here so that cache can be shared
                        frame_list.append(os.path.join(video, frame_name))
                if len(frame_list) >= min_len:
                    video_list.append(frame_list)
                    length_list.append(len(frame_list))
                frame_list = frames = None  # empty
            self.videos, self.lengths = video_list, length_list
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump({'root': self.root,
                                 'videos': self.videos,
                                 'lengths': self.lengths}, f)
        self.cumsum = np.cumsum([0] + self.lengths)
        self.lengthsn = [i - nframe_num + 1 for i in self.lengths]
        self.cumsumn = np.cumsum([0] + self.lengthsn)
        print("Total number of videos {}.".format(len(self.videos)))
        print("Total number of frames {}.".format(np.sum(self.lengths)))
        if self.mode == 'video':
            self._dataset_length = len(self.videos)
        elif self.mode == 'image':
            self._dataset_length = np.sum(self.lengths)
        elif self.mode == 'nframe':
            self._dataset_length = np.sum(self.lengthsn)
        else:
            raise NotImplementedError

    def _get_video(self, index):
        video_len = self.lengths[index]
        start_idx = random.randint(0, video_len-self.frame_num*self.frame_step)
        frames = []
        for i in range(start_idx, start_idx+self.frame_num*self.frame_step, self.frame_step):
            img = Image.open(os.path.join(self.root, self.videos[index][i]))
            frames.append(F.to_tensor(img))
        frames = torch.stack(frames, 0)
        frames = self.transform(frames)
        return {'frames': frames, 'path': os.path.basename(os.path.dirname(self.videos[index][0]))}

    def _get_image(self, index):
        # copied from MoCoGAN
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsum, index) - 1
            frame_id = index - self.cumsum[video_id] - 1
        frame = Image.open(os.path.join(self.root, self.videos[video_id][frame_id]))
        frame = F.to_tensor(frame)
        frame = self.transform(frame)  # no ToTensor in transform
        return frame

    def _get_nframe(self, index):
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsumn, index) - 1
            frame_id = index - self.cumsumn[video_id] - 1
        frames = []
        for i in range(self.nframe_num):
            frame = Image.open(os.path.join(self.root, self.videos[video_id][frame_id + i]))
            frames.append(F.to_tensor(frame))
        frames = torch.stack(frames, 0)
        frames = self.transform(frames)
        return frames.unbind(0) if self.unbind else frames

    def __getitem__(self, index):
        if self.mode == 'video':
            return self._get_video(index)
        elif self.mode == 'image':
            return self._get_image(index)
        elif self.mode == 'nframe':
            return self._get_nframe(index)
        else:
            return None

    def __len__(self):
        return self._dataset_length


def get_image_dataset(args, which_dataset='c10', data_root='./data', train=True):
    # Define Image Datasets (VideoFolder will be the collection of all frames)
    CropLongEdge = RandomCropLongEdge if train else CenterCropLongEdge
    random_flip = getattr(args, 'flip', True)
    dataset = None
    if which_dataset.lower() in ['multires']:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5 if random_flip else 0),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        dataset = MultiResolutionDataset(data_root, transform, args.size)
    elif which_dataset.lower() in ['videofolder']:
        # [Note] Potentially, same transforms will be applied to a batch of images,
        # either a sequence or a pair (optical flow), so we should apply ToTensor first.
        transform = transforms.Compose(
            [
                # transforms.ToTensor(),  # this should be done in loader
                transforms.RandomHorizontalFlip(p=0.5 if random_flip else 0),
                transforms.Resize(args.size),  # Image.LANCZOS
                transforms.CenterCrop(args.size),
                # transforms.ToTensor(),  # normally placed here
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        dataset = VideoFolderDataset(data_root, transform, mode='image', cache=args.cache)
    elif which_dataset.lower() in ['imagefolder', 'custom']:
        transform = transforms.Compose(
            [
                CropLongEdge(),
                transforms.Resize(args.size, Image.LANCZOS),
                transforms.RandomHorizontalFlip(p=0.5 if random_flip else 0),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = torchvision.datasets.ImageFolder(
            root=data_root,
            transform=transform
        )
    elif which_dataset.lower() in ['mnist']:
        transform = transforms.Compose(
            [
                transforms.Resize(args.size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = torchvision.datasets.MNIST(
            root=data_root, download=True,
            train=train,
            transform=transform
        )
    elif which_dataset.lower() in ['cifar10', 'c10']:
        transform = transforms.Compose(
            [
                transforms.Resize(args.size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = torchvision.datasets.CIFAR10(
            root=data_root, download=True,
            train=train,
            transform=transform
        )
    elif which_dataset.lower() in ['cifar100', 'c100']:
        transform = transforms.Compose(
            [
                transforms.Resize(args.size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = torchvision.datasets.CIFAR100(
            root=data_root, download=True,
            train=train,
            transform=transform
        )
    elif which_dataset.lower() in ['imagenet', 'ilsvrc2012']:
        # TODO: save file index, hdf5 or lmdb
        transform = transforms.Compose(
            [
                CropLongEdge(),
                transforms.Resize(args.size),
                transforms.RandomHorizontalFlip(p=0.5 if random_flip else 0),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_root, 'train' if train else 'valid'),
            transform=transform
        )
    elif which_dataset.lower() in ['tiny_imagenet', 'tiny']:
        transform = transforms.Compose(
            [
                transforms.Resize(args.size),
                transforms.RandomHorizontalFlip(p=0.5 if random_flip else 0),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_root, 'train' if train else 'test'),
            transform=transform
        )
    else:
        raise NotImplementedError
    return dataset
