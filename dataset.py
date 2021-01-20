from io import BytesIO

import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import pickle
import os
import numpy as np
import tqdm
import random

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


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
        cache=None,
    ):
        assert(mode in ['video', 'image', 'pair', 'triplet'])
        self.mode = mode
        self.root = dataroot
        self.cache = cache
        self.transform = transform or transforms.ToTensor()
        self.min_len = min_len
        self.frame_num = frame_num
        self.frame_step = frame_step
        self.videos = []
        self.lengths = []
        
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
                    frames = sorted(os.listdir(os.path.join(dataroot, video)))
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
        self.lengths1 = [i - 1 for i in self.lengths]
        self.cumsum1 = np.cumsum([0] + self.lengths1)
        self.lengths2 = [i - 2 for i in self.lengths]
        self.cumsum2 = np.cumsum([0] + self.lengths2)
        print("Total numver of videos {}.".format(len(self.videos)))
        print("Total number of frames {}.".format(np.sum(self.lengths)))
        if self.mode == 'video':
            self._dataset_length = len(self.videos)
        elif self.mode == 'image':
            self._dataset_length = np.sum(self.lengths)
        elif self.mode == 'pair':
            self._dataset_length = np.sum(self.lengths1)
        else:  # self.mode == 'triplet'
            self._dataset_length = np.sum(self.lengths2)

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
    
    def _get_pair(self, index):
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsum1, index) - 1
            frame_id = index - self.cumsum1[video_id] - 1
        frame1 = Image.open(os.path.join(self.root, self.videos[video_id][frame_id]))
        frame1 = F.to_tensor(frame1)
        frame2 = Image.open(os.path.join(self.root, self.videos[video_id][frame_id + 1]))
        frame2 = F.to_tensor(frame2)
        # We should apply identical transforms to frame1 & frame2
        frames = torch.stack([frame1, frame2], 0)
        frames = self.transform(frames)
        return frames.unbind(0)
    
    def _get_triplet(self, index):
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsum2, index) - 1
            frame_id = index - self.cumsum2[video_id] - 1
        frame1 = Image.open(os.path.join(self.root, self.videos[video_id][frame_id]))
        frame1 = F.to_tensor(frame1)
        frame2 = Image.open(os.path.join(self.root, self.videos[video_id][frame_id + 1]))
        frame2 = F.to_tensor(frame2)
        frame3 = Image.open(os.path.join(self.root, self.videos[video_id][frame_id + 2]))
        frame3 = F.to_tensor(frame3)
        frames = torch.stack([frame1, frame2, frame3], 0)
        frames = self.transform(frames)
        return frames.unbind(0)
    
    def __getitem__(self, index):
        if self.mode == 'video':
            return self._get_video(index)
        elif self.mode == 'image':
            return self._get_image(index)
        elif self.mode == 'pair':
            return self._get_pair(index)
        else:  # mode == 'triplet'
            return self._get_triplet(index)

    def __len__(self):
        return self._dataset_length
