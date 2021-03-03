import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2
import random
import pickle
import pdb
st = pdb.set_trace

_IMGEXT = 'png'

def get_frames(vidpath, image_size=0, every_nth=1, trim_len=float('Inf')):
    # get frames as list of images, return a list of list
    vidcap = cv2.VideoCapture(vidpath)
    success, image = vidcap.read()  # image is of shape [H, W, C]
    clips = []
    idx = 0
    count = 0
    images = []
    while success:
        if idx % every_nth == 0:
            if image_size > 0 and (image.shape[0] != image_size or image.shape[1] != image_size):
                image = cv2.resize(image, (image_size, image_size))
            images.append(image)
            count += 1
        if count >= trim_len:
            clips.append(images)
            count = 0
            images = []
        success, image = vidcap.read()
        idx += 1
    if len(clips) == 0:
        clips = [images]
    return clips


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=0)
    parser.add_argument('--raw_data_root', type=str, default='../data/action/videos')
    parser.add_argument('--dest_data_root', type=str, default='../data/action/frames')
    parser.add_argument('--trim_len', type=float, default=float('Inf'))
    parser.add_argument('--every_nth', type=int, default=1)
    parser.add_argument('--cache', type=str, default=None)
    args = parser.parse_args()

    # sys.path.append('..')
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    if not os.path.exists(args.dest_data_root):
        os.mkdir(args.dest_data_root)
    videopaths = []  # a list of list
    videos = os.listdir(args.raw_data_root)
    for video in videos:
        mp4path = os.path.join(args.raw_data_root, video)
        videoname = video.rstrip('.avi')
        clips = get_frames(mp4path, args.image_size, args.every_nth, args.trim_len)
        idx_clip = range(len(clips))
        for j in idx_clip:
            clippath = os.path.join(args.dest_data_root, f'{videoname}_{j:02d}')
            if not os.path.exists(clippath):
                os.mkdir(clippath)
            clippaths = []
            for k, img in enumerate(clips[j]):
                imgpath = os.path.join(clippath, f'{k:07d}.{_IMGEXT}')
                clippaths.append(os.path.join(f'{videoname}_{j:02d}', f'{k:07d}.{_IMGEXT}'))
                cv2.imwrite(imgpath, img)
            videopaths.append(clippaths)
        print(f'=> {video}')
    
    if args.cache:
        with open(args.cache, 'wb') as f:
            pickle.dump(videopaths, f)
