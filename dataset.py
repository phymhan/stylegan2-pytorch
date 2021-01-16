from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import os
import numpy as np
import tqdm


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
    def __init__(self, dataroot, transform=None, mode='video', min_len=8, cache=None):
        self.root = dataroot
        self.transform = transform or transforms.ToTensor()
        self.cache = cache
        self.mode = mode
        self.min_len = min_len
        self.videos = []
        self.lengths = []
        assert(mode in ['video', 'image', 'pair'])
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.videos, self.lengths = pickle.load(f)
        else:
            video_list = []
            length_list = []
            for i, video in enumerate(tqdm.tqdm(os.listdir(dataroot), desc="Counting videos")):
                try:
                    frames = sorted(os.listdir(os.path.join(dataroot, video)))
                except ValueError:
                    continue
                frame_list = []
                for j, frame_name in enumerate(frames):
                    if is_image_file(os.path.join(dataroot, video, frame_name)):
                        # do not include dataroot here so that cache can be shared
                        frame_list.append(os.path.join(video, frame_name))
                if len(frame_list) >= min_len:
                    video_list.append(frame_list)
                    length_list.append(len(frame_list))
            self.videos, self.lengths = video_list, length_list
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump((self.videos, self.lengths), f)
        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total numver of videos {}.".format(len(self.videos)))
        print("Total number of frames {}.".format(np.sum(self.lengths)))

    def _get_video(self, index):
        # copied from Yu
        video = self.videos[index]
        video_len = self.lengths[index]
        n_frames = 50
        
        start_idx = random.randint(0, video_len-1-n_frames*FRAME_STEP)
        img = Image.open(video[0])
        h, w = img.height, img.width
        
        NEED_CROP = False
        if h > w:
            NEED_CROP = True
            half = (h-w) // 2
            cropsize = (0, half, w, half+w) # left, upper, right, lower
        elif w > h:
            NEED_CROP = True
            half = (w-h) // 2
            cropsize = (half, 0, half+h, h)

        images = []
        
        for i in range(start_idx, start_idx+n_frames*FRAME_STEP, FRAME_STEP):
            path = video[i]
            img = Image.open(path)
            if NEED_CROP:
                img = img.crop(cropsize)
            if img.height != img.width:
                print('crop error!')
                abc = 1
            img = img.resize(self.opt.fineSize, Image.ANTIALIAS)
            img = np.asarray(img, dtype=np.float32)
            img /= 255.
            img_tensor = preprocess(img).unsqueeze(0)
            images.append(img_tensor)
        
        video_clip = torch.cat(images)
        return video_clip
    
    def _get_image(self, index):
        # copied from MoCoGAN
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsum, index) - 1
            frame_id = index - self.cumsum[video_id] - 1
        frame = Image.open(self.videos[video_id][frame_id])
        frame = self.transform(frame)
        return frame
    
    def __getitem__(self, index):
        if self.mode == 'video':
            return self._get_video(index)
        elif self.mode == 'image':
            return self._get_image(index)
        else:  # 'pair'
            return None


    def __len__(self):
        return len(self.videos) if self.mode == 'video' else np.sum(self.lengths)
