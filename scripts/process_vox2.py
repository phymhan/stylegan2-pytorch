import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2
import random
import pickle

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
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--raw_data_root', type=str, default='../data/voxceleb/dev/mp4')
    parser.add_argument('--dest_data_root', type=str, default='../data/vox')
    parser.add_argument('--num_identities', type=int, default=100)
    parser.add_argument('--trim_len', type=float, default=float('Inf'))
    parser.add_argument('--every_nth', type=int, default=1)
    parser.add_argument('--num_utterance_per_video', type=int, default=1)
    parser.add_argument('--num_clips_per_utterance', type=int, default=1)
    parser.add_argument('--cache', type=str, default='')
    args = parser.parse_args()

    # sys.path.append('..')
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    if not os.path.exists(args.dest_data_root):
        os.mkdir(args.dest_data_root)
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
    
    videopaths = []  # a list of list
    for id in ids:
        videos = os.listdir(os.path.join(args.raw_data_root, id))
        for video in videos:
            if video.endswith('.png'):
                continue
            utterances = os.listdir(os.path.join(args.raw_data_root, id, video))
            idx_utterance = range(len(utterances))
            idx_utterance = np.sort(np.random.choice(idx_utterance, min(len(utterances), args.num_utterance_per_video), False))
            for i in idx_utterance:
                utterance = utterances[i]
                mp4path = os.path.join(args.raw_data_root, id, video, utterance)
                clips = get_frames(mp4path, args.image_size, args.every_nth, args.trim_len)
                # clips = [[frames_for_clip_1], ...]
                idx_clip = range(len(clips))
                idx_clip = np.sort(np.random.choice(idx_clip, min(len(clips), args.num_clips_per_utterance), False))
                for j in idx_clip:
                    clippath = os.path.join(args.dest_data_root, f'{id}-{video}-{utterance}_{j:02d}')
                    if not os.path.exists(clippath):
                        os.mkdir(clippath)
                    clippaths = []
                    for k, img in enumerate(clips[j]):
                        imgpath = os.path.join(clippath, f'{k:07d}.{_IMGEXT}')
                        clippaths.append(os.path.join(f'{id}-{video}-{utterance}_{j:02d}', f'{k:07d}.{_IMGEXT}'))
                        cv2.imwrite(imgpath, img)
                    videopaths.append(clippaths)
            print(f'=> {id} {video}')
    
    if args.cache:
        with open(args.cache, 'wb') as f:
            pickle.dump(videopaths, f)
