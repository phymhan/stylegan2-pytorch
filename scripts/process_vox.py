import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2
import random


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
    return clips


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--raw_data_root', type=str, default='../data/voxceleb/dev/mp4')
    parser.add_argument('--dest_data_root', type=str, default='../data/vox')
    parser.add_argument('--num_identities', type=int, default=100)
    parser.add_argument('--trim_len', type=float, default=float('Inf'))
    parser.add_argument('--every_nth', type=int, default=1)
    parser.add_argument('--num_clips_per_video', type=int, default=2)
    args = parser.parse_args()

    # sys.path.append('..')
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    ids_file = os.path.join(args.dest_data_root, f'id_{args.num_identities}_seed_{seed}.txt')
    if os.path.exists(ids_file):
        with open(ids_file, 'r') as f:
            ids = [id.strip() for id in f.readlines()]
    else:
        ids = os.listdir(args.raw_data_root)
        ids = random.choices(ids, k=args.num_identities)
        ids.sort()
        with open(ids_file, 'w') as f:
            for id in ids:
                f.write(f'{id}\n')
    
    for id in ids:
        folders = os.listdir(os.path.join(args.raw_data_root, id))
        if not os.path.exists(os.path.join(args.dest_data_root, id)):
            os.mkdir(os.path.join(args.dest_data_root, id))
        for folder in folders:
            if folder.endswith('.png'):
                continue
            videos = os.listdir(os.path.join(args.raw_data_root, id, folder))
            for video in videos:
                vidpath = os.path.join(args.raw_data_root, id, folder, video)
                clips = get_frames(vidpath, args.image_size, args.every_nth, args.trim_len)
                # clips = [[frames_for_clip_1], ...]
                idx_clip = range(len(clips))
                idx_clip = np.sort(np.random.choice(idx_clip, min(len(clips), args.num_clips_per_video), False))
                for j in idx_clip:
                    clippath = os.path.join(args.dest_data_root, id, f'{folder}-{video}_{j:02d}')
                    if not os.path.exists(clippath):
                        os.mkdir(clippath)
                    for k, img in enumerate(clips[j]):
                        imgpath = os.path.join(clippath, f'{k:07d}.jpg')
                        cv2.imwrite(imgpath, img)
                print(f'=> video {vidpath}')
