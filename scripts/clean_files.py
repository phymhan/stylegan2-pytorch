import os
import sys
import shutil
import argparse


def clean_experiment(exp_path, clean_sample=False, clean_weight=True, save_every='10000'):
    folders = []
    if clean_sample:
        folders.append('sample')
    if clean_weight:
        folders.append('weight')
    for folder in folders:
        files = os.listdir(os.path.join(exp_path, folder))
        files = sorted(files)
        keep_files = []
        if 'latest' in files[-1]:
            keep_files = [files[-1]]
            files = files[:-1]
        if save_every == 'auto':
            percentile = [0.5, 0.75, 1]
            keep_idx = [int((len(files)-1)*p) for p in percentile]
            keep_files += [files[i] for i in set(keep_idx)]
        remove_files = set(files) - set(keep_files)
        for i in remove_files:
            os.remove(os.path.join(exp_path, folder, i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root', type=str, default='logs')
    parser.add_argument('--name', type=str, nargs='+', default=[])
    parser.add_argument('--folder', type=str, default='weight', help='sample, weight')
    parser.add_argument('--save_every', type=str, default='auto', help='1000, 10000, auto')
    args = parser.parse_args()
    root = os.path.join(os.path.dirname(sys.argv[0]), os.pardir)

    if len(args.name) == 0:
        names = os.listdir(os.path.join(root, args.log_root))
    for name in names:
        clean_experiment(os.path.join(root, args.log_root, name), args.folder=='sample', args.folder=='weight', args.save_every)
