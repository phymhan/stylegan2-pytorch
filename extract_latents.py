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


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    # Endless iterator
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


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

    for data in tqdm(loader):
        real_seq = data['frames']
        real_seq = real_seq.to(device)  # shape [1, T, 3, H, W]
        latent_seq = encoder(real_seq.squeeze())  # shape [T, n_latent, 512]
        latent_npy = latent_seq.detach().cpu().numpy()
        
        np.save(os.path.join(args.output_dir, f"{data['path']}.npy"), latent_npy)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 encoder trainer")

    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--cache", type=str, default='local.db')
    parser.add_argument("--name", type=str, help="experiment name", default='default_exp')
    parser.add_argument("--output_dir", type=str, default='samples/latents')
    parser.add_argument("--use_wscale", action='store_true', help="whether to use `wscale` layer in idinvert encoder")
    parser.add_argument("--train_on_fake", action='store_true', help="train encoder on fake?")
    parser.add_argument("--which_encoder", type=str, default='style')
    parser.add_argument("--which_latent", type=str, default='w_shared')
    parser.add_argument("--frame_num", type=int, default=50)
    parser.add_argument("--frame_step", type=int, default=1)
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
    if args.which_latent == 'w':
        args.latent_full = args.latent * args.n_latent
    elif args.which_latent == 'w_shared':
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
            which_latent=args.which_latent, reshape_latent=True).to(device)
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
    loader = data.DataLoader(
        dataset,
        batch_size=1,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    run(args, loader, e_ema, g_ema, device)
