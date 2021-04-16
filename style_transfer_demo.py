import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from torchvision.io import write_video
from PIL import Image
from tqdm import tqdm
import util
import pdb
st = pdb.set_trace

try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator
from idinvert_pytorch.models.perceptual_model import VGG16
from dataset import MultiResolutionDataset, VideoFolderDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment, AdaptiveAugment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(loader):
    # Endless iterator
    while True:
        for batch in loader:
            yield batch


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def accumulate_batches(data_iter, num):
    samples = []
    while num > 0:
        data = next(data_iter)
        samples.append(data)
        num -= data.size(0)
    samples = torch.cat(samples, dim=0)
    if num < 0:
        samples = samples[:num, ...]
    return samples


def adjust_batch(data, group=1):
    # Adjust batch size to multiple of group
    # data is of shape [N, C, H, W]
    batch = data.shape[0]
    if batch % group == 0:
        return data
    batch_new = int(math.ceil(batch / group * 1.0) * group)
    repeat_dims = [int(math.ceil(batch_new / batch * 1.0))] + [1] * (data.ndim - 1)
    return data.repeat(*repeat_dims)


def encode(encoder, seq, args):
    # seq is of shape [T, C, H, W]
    T = seq.shape[0]
    if T % args.stddev_group != 0:
        seq = adjust_batch(seq, args.stddev_group)
        latent = encoder(seq)  # shape [N, n_latent, 512]
        latent = latent[:T, ...]
    else:
        latent = encoder(seq)  # shape [T, n_latent, 512]
    return latent


@torch.no_grad()
def run(args, loader, encoder, generator, device):
    if args.distributed:
        e_module = encoder.module
        g_module = generator.module
    else:
        e_module = encoder
        g_module = generator

    requires_grad(encoder, False)
    requires_grad(generator, False)
    encoder.eval()
    generator.eval()
    counter = 0

    for data in tqdm(loader):
        counter += 1
        real_seq = data['frames']
        real_seq = real_seq.to(device)  # shape [2, T, 3, H, W]
        seqA, seqB = real_seq[0, ...], real_seq[1, ...]
        latA = encode(encoder, seqA, args)
        latB = encode(encoder, seqB, args)
        coA, moA = latA[0, ...], latA[1:, ...] - latA[0, ...]
        coB, moB = latB[0, ...], latB[1:, ...] - latB[0, ...]

        # recon
        recA, _ = generator([coA + moA], input_is_latent=True, return_latents=False)
        recB, _ = generator([coB + moB], input_is_latent=True, return_latents=False)

        # swap
        coAmoB, _ = generator([coA + moB], input_is_latent=True, return_latents=False)
        coBmoA, _ = generator([coB + moA], input_is_latent=True, return_latents=False)

        # orig
        oriA = seqA[1:, ...]
        oriB = seqB[1:, ...]

        # save images and videos
        T = seqA.shape[0]
        utils.save_image(oriA, os.path.join(args.output_dir, f'{counter:03d}-img_origA.png'), nrow=T-1, normalize=True, range=(-1, 1))
        utils.save_image(oriB, os.path.join(args.output_dir, f'{counter:03d}-img_origB.png'), nrow=T-1, normalize=True, range=(-1, 1))
        utils.save_image(recA, os.path.join(args.output_dir, f'{counter:03d}-img_reconA.png'), nrow=T-1, normalize=True, range=(-1, 1))
        utils.save_image(recB, os.path.join(args.output_dir, f'{counter:03d}-img_reconB.png'), nrow=T-1, normalize=True, range=(-1, 1))
        utils.save_image(coAmoB, os.path.join(args.output_dir, f'{counter:03d}-img_coA+moB.png'), nrow=T-1, normalize=True, range=(-1, 1))
        utils.save_image(coBmoA, os.path.join(args.output_dir, f'{counter:03d}-img_coB+moA.png'), nrow=T-1, normalize=True, range=(-1, 1))
        util.save_video(oriA, os.path.join(args.output_dir, f'{counter:03d}-vid_origA.mp4'))
        util.save_video(oriB, os.path.join(args.output_dir, f'{counter:03d}-vid_origB.mp4'))
        util.save_video(recA, os.path.join(args.output_dir, f'{counter:03d}-vid_reconA.mp4'))
        util.save_video(recB, os.path.join(args.output_dir, f'{counter:03d}-vid_reconB.mp4'))
        util.save_video(coAmoB, os.path.join(args.output_dir, f'{counter:03d}-vid_coA+moB.mp4'))
        util.save_video(coBmoA, os.path.join(args.output_dir, f'{counter:03d}-vid_coB+moA.mp4'))
        if counter >= args.n_sample:
            break


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 encoder trainer")

    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--cache", type=str, default='local.db')
    parser.add_argument("--name", type=str, help="experiment name", default='default_exp')
    parser.add_argument("--output_dir", type=str, default='samples/style_transfer')
    parser.add_argument("--use_wscale", action='store_true', help="whether to use `wscale` layer in idinvert encoder")
    parser.add_argument("--train_on_fake", action='store_true', help="train encoder on fake?")
    parser.add_argument("--which_encoder", type=str, default='style')
    parser.add_argument("--which_latent", type=str, default='w_tied')
    parser.add_argument("--frame_num", type=int, default=50)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--stddev_group", type=int, default=4)
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.n_latent = int(np.log2(args.size)) * 2 - 2  # used in Generator
    args.latent = 512  # fixed, dim of w or z (same size)
    if args.which_latent == 'w_plus':
        args.latent_full = args.latent * args.n_latent
    elif args.which_latent == 'w_tied':
        args.latent_full = args.latent
    else:
        raise NotImplementedError
    args.n_mlp = 8

    args.start_iter = 0
    # util.set_log_dir(args)
    # util.print_args(parser, args)
    
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()

    if args.which_encoder == 'idinvert':
        from idinvert_pytorch.models.stylegan_encoder_network import StyleGANEncoderNet
        e_ema = StyleGANEncoderNet(resolution=args.size, w_space_dim=args.latent,
            which_latent=args.which_latent, reshape_latent=True,
            use_wscale=args.use_wscale).to(device)
    else:
        from model import Encoder
        e_ema = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
            which_latent=args.which_latent, reshape_latent=True, stddev_group=args.stddev_group).to(device)
    e_ema.eval()

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        g_ema.load_state_dict(ckpt["g_ema"])
        e_ema.load_state_dict(ckpt["e_ema"])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    dataset = VideoFolderDataset(args.path, transform, mode='video', cache=args.cache,
                                 frame_num=args.frame_num, frame_step=args.frame_step)
    if len(dataset) == 0:
        raise ValueError
    loader = data.DataLoader(
        dataset,
        batch_size=2,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.stddev_group = min(args.stddev_group, args.batch)
    run(args, loader, e_ema, g_ema, device)
