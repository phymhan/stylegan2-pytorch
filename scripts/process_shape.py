import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2
import random
import pickle

_IMGEXT = 'jpg'

def get_frames(vidpath, image_size=0, trim_len=float('Inf')):
    # get frames as list of images, return a list of list
    video = cv2.imread(vidpath)
    horizontal = video.shape[1] > video.shape[0]
    shorter, longer = min(video.shape[0], video.shape[1]), max(video.shape[0], video.shape[1])
    video_len = longer // shorter
    frames = np.split(video, video_len, axis=1 if horizontal else 0)
    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=0)
    parser.add_argument('--raw_data_root', type=str, default='/research/cbim/vast/lh599/data/moving_shapes/shapes64/train/videos')
    parser.add_argument('--dest_data_root', type=str, default='../data/shape64/train/videos')
    parser.add_argument('--trim_len', type=float, default=float('Inf'))
    parser.add_argument('--cache', type=str, default='')
    args = parser.parse_args()

    # sys.path.append('..')
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(args.dest_data_root):
        os.makedirs(args.dest_data_root)
    
    videopaths = []  # a list of list
    videos = os.listdir(os.path.join(args.raw_data_root))
    for video in videos:
        framepaths = []
        videoname = video.rstrip('.png').rstrip('.jpg')
        if not os.path.exists(os.path.join(args.dest_data_root, videoname)):
            os.mkdir(os.path.join(args.dest_data_root, videoname))
        videopath = os.path.join(args.raw_data_root, video)
        frames = get_frames(videopath)
        for k, img in enumerate(frames):
            imgpath = os.path.join(args.dest_data_root, videoname, f'{k:07d}.{_IMGEXT}')
            framepaths.append(os.path.join(videoname, f'{k:07d}.{_IMGEXT}'))
            cv2.imwrite(imgpath, img)
        videopaths.append(framepaths)
    if args.cache:
        with open(args.cache, 'wb') as f:
            pickle.dump(videopaths, f)
