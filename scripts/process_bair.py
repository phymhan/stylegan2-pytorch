import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2
import random
import pickle
import natsort
import shutil
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
    parser.add_argument('--raw_data_root', type=str, default='../data/bair/processed_data/train')
    parser.add_argument('--dest_data_root', type=str, default='../data/bair/subsets')
    parser.add_argument('--every_nth', type=int, default=1)
    parser.add_argument('--cache', type=str, default=None)
    parser.add_argument('--num_traj', type=int, default=50)
    args = parser.parse_args()

    # sys.path.append('..')
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    if not os.path.exists(args.dest_data_root):
        os.mkdir(args.dest_data_root)
    ids_file = os.path.join(args.dest_data_root, f'traj_{args.num_traj}_seed_{seed}.txt')
    if os.path.exists(ids_file):
        with open(ids_file, 'r') as f:
            traj = [i.strip() for i in f.readlines()]
    else:
        traj = os.listdir(args.raw_data_root)
        traj = random.choices(traj, k=args.num_traj)
        with open(ids_file, 'w') as f:
            for i in traj:
                f.write(f'{i}\n')

    videopaths = []  # a list of list
    for tr in traj:
        videos = os.listdir(os.path.join(args.raw_data_root, tr))
        for video in videos:
            mp4path = os.path.join(args.raw_data_root, tr, video)
            frames = natsort.natsorted(os.listdir(mp4path))
            clippath = os.path.join(args.dest_data_root, f'{tr}_{video}')
            if not os.path.exists(clippath):
                os.mkdir(clippath)
            clippaths = []
            for fr in frames:
                shutil.copyfile(
                    os.path.join(args.raw_data_root, tr, video, fr),
                    os.path.join(args.dest_data_root, f'{tr}_{video}', fr)
                )
                clippaths.append(os.path.join(args.dest_data_root, f'{tr}_{video}', fr))
            videopaths.append(clippaths)
            print(f'=> {tr} {video}')
    
    if args.cache:
        with open(args.cache, 'wb') as f:
            pickle.dump(videopaths, f)
