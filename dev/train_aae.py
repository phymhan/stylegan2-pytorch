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
from torchvision import datasets, transforms, utils
from PIL import Image
from tqdm import tqdm
import util
from calc_inception import load_patched_inception_v3
from fid import extract_feature_from_samples, calc_fid
import pickle
import pdb
st = pdb.set_trace

try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator
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


def sample_data2(loader):
    # image and label pair
    while True:
        for batch, _ in loader:
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
        imgs = next(data_iter)
        samples.append(imgs)
        num -= imgs.size(0)
    samples = torch.cat(samples, dim=0)
    if num < 0:
        samples = samples[:num, ...]
    return samples


def load_real_samples(args, data_iter):
    # if args.cache is not None:
    #     npy_path = os.path.splitext(args.cache)[0] + f"_real_{args.n_sample}.npy"
    # else:
    #     npy_path = None
    npy_path = args.sample_cache
    if npy_path is not None and os.path.exists(npy_path):
        sample_x = torch.from_numpy(np.load(npy_path)).to(args.device)
    else:
        sample_x = accumulate_batches(data_iter, args.n_sample).to(args.device)
        if npy_path is not None:
            np.save(npy_path, sample_x.cpu().numpy())
    return sample_x


def train(args, loader, generator, encoder, discriminator, prior, vggnet, g_optim, e_optim, d_optim, p_optim, g_ema, e_ema, device):
    if args.dataset == 'imagefolder':
        loader = sample_data2(loader)
    else:
        loader = sample_data(loader)
    
    if args.eval_every > 0:
        inception = nn.DataParallel(load_patched_inception_v3()).to(device)
        inception.eval()
        with open(args.inception, "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]
    else:
        inception = real_mean = real_cov = None
    mean_latent = None

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}
    avg_pix_loss = util.AverageMeter()
    avg_vgg_loss = util.AverageMeter()

    if args.distributed:
        g_module = generator.module
        e_module = encoder.module
        d_module = discriminator.module
    else:
        g_module = generator
        e_module = encoder
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 256, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    sample_x = load_real_samples(args, loader)
    if sample_x.ndim > 4:
        sample_x = sample_x[:,0,...]

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)

        # Train Discriminator
        requires_grad(generator, False)
        requires_grad(encoder, False)
        requires_grad(prior, False)
        requires_grad(discriminator, True)

        latent_real, _ = encoder(real_img)
        noise = torch.randn(args.batch, args.latent, device=args.device)
        latent_fake = prior(noise)
        fake_pred = discriminator(latent_fake)
        real_pred = discriminator(latent_real)
        d_loss = d_logistic_loss(real_pred, fake_pred)
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()
        loss_dict["d"] = d_loss

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            latent_real = latent_real.detach()
            latent_real.requires_grad = True
            real_pred = discriminator(latent_real)
            r1_loss = d_r1_loss(real_pred, latent_real)
            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()
        loss_dict["r1"] = r1_loss

        # Train Encoder and Generator
        requires_grad(encoder, True)
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(prior, False)

        latent_real, _ = encoder(real_img)
        rec_img, _ = generator([latent_real], input_is_latent=True, return_latents=False)
        if args.lambda_pix > 0:
            if args.pix_loss == 'l2':
                pix_loss = torch.mean((rec_img - real_img) ** 2)
            else:
                pix_loss = F.l1_loss(rec_img, real_img)
        if args.lambda_vgg > 0:
            vgg_loss = torch.mean((vggnet(real_img) - vggnet(rec_img)) ** 2)

        e_loss = pix_loss * args.lambda_pix + vgg_loss * args.lambda_vgg

        encoder.zero_grad()
        generator.zero_grad()
        e_loss.backward()
        # generator.style.zero_grad()  # do not update F (or generator.style)
        e_optim.step()
        g_optim.step()

        loss_dict["e"] = e_loss
        
        pix_loss_val = pix_loss.mean().item()
        vgg_loss_val = vgg_loss.mean().item()

        accumulate(e_ema, e_module, accum)
        accumulate(g_ema, g_module, accum)

        # Train Prior
        requires_grad(generator, False)
        requires_grad(encoder, False)
        requires_grad(prior, True)
        requires_grad(discriminator, False)

        noise = torch.randn(args.batch, args.latent, device=args.device)
        latent_fake = prior(noise)
        fake_pred = discriminator(latent_fake)
        p_loss = g_nonsaturating_loss(fake_pred)
        loss_dict["p"] = p_loss

        prior.zero_grad()
        p_loss.backward()
        p_optim.step()

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        # g_loss_val = loss_reduced["g"].mean().item()
        e_loss_val = loss_reduced["e"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        # path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        # path_length_val = loss_reduced["path_length"].mean().item()
        avg_pix_loss.update(pix_loss_val, real_img.shape[0])
        avg_vgg_loss.update(vgg_loss_val, real_img.shape[0])

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; e: {e_loss_val:.4f}; r1: {r1_val:.4f}; "
                    # f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if i % args.log_every == 0:
                with torch.no_grad():
                    latent_x, _ = e_ema(sample_x)
                    fake_x, _ = generator([latent_x], input_is_latent=True, return_latents=False)
                    sample_pix_loss = torch.sum((sample_x - fake_x) ** 2)
                with open(os.path.join(args.log_dir, 'log.txt'), 'a+') as f:
                    f.write(f"{i:07d}; pix: {avg_pix_loss.avg}; vgg: {avg_vgg_loss.avg}; "
                            f"ref: {sample_pix_loss.item()};\n")

            if args.eval_every > 0 and i % args.eval_every == 0:
                with torch.no_grad():
                    g_ema.eval()
                    if args.truncation < 1:
                        # mean_latent = g_ema.mean_latent(4096)
                        latent_in = torch.randn(4096, 512, device=args.device)
                        mean_latent = prior(latent_in).mean(0, keepdim=True)
                    features = extract_feature_from_samples(
                        g_ema, inception, args.truncation, mean_latent, 64, args.n_sample_fid, args.device, prior=prior
                    ).numpy()
                    sample_mean = np.mean(features, 0)
                    sample_cov = np.cov(features, rowvar=False)
                    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
                print("fid:", fid)
                with open(os.path.join(args.log_dir, 'log_fid.txt'), 'a+') as f:
                    f.write(f"{i:07d}: fid: {float(fid):.4f}\n")

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % args.log_every == 0:
                with torch.no_grad():
                    # Fixed fake samples
                    g_ema.eval()
                    # sample, _ = g_ema([sample_z])
                    sample_w = prior(sample_z)
                    sample, _ = g_ema([sample_w], input_is_latent=True)
                    utils.save_image(
                        sample,
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-sample.png"),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    # Reconstruction samples
                    e_ema.eval()
                    nrow = int(args.n_sample ** 0.5)
                    nchw = list(sample_x.shape)[1:]
                    latent_real, _ = e_ema(sample_x)
                    fake_img, _ = g_ema([latent_real], input_is_latent=True, return_latents=False)
                    sample = torch.cat((sample_x.reshape(args.n_sample//nrow, nrow, *nchw), 
                                        fake_img.reshape(args.n_sample//nrow, nrow, *nchw)), 1)
                    utils.save_image(
                        sample.reshape(2*args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-recon.png"),
                        nrow=nrow,
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % args.save_every == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "e": e_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "e_ema": e_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                        "iter": i,
                    },
                    os.path.join(args.log_dir, 'weight', f"{str(i).zfill(6)}.pt"),
                )
            
            if i % args.save_latest_every == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "e": e_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "e_ema": e_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                        "iter": i,
                    },
                    os.path.join(args.log_dir, 'weight', f"latest.pt"),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--dataset", type=str, default='multires')
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--sample_cache", type=str, default=None)
    parser.add_argument("--name", type=str, help="experiment name", default='default_exp')
    parser.add_argument("--log_root", type=str, help="where to save training logs", default='logs')
    parser.add_argument("--log_every", type=int, default=100, help="save samples every # iters")
    parser.add_argument("--save_every", type=int, default=1000, help="save checkpoints every # iters")
    parser.add_argument("--save_latest_every", type=int, default=100, help="save latest checkpoints every # iters")
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
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
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=1,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
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
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
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
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--which_encoder", type=str, default='style')
    parser.add_argument("--which_latent", type=str, default='w_tied')
    parser.add_argument("--stddev_group", type=int, default=1)
    parser.add_argument("--use_wscale", action='store_true', help="whether to use `wscale` layer in idinvert encoder")
    parser.add_argument("--vgg_ckpt", type=str, default="vgg16.pth")
    parser.add_argument("--output_layer_idx", type=int, default=23)
    parser.add_argument("--lambda_vgg", type=float, default=5e-5)
    parser.add_argument("--lambda_adv", type=float, default=0.1)
    parser.add_argument("--lambda_pix", type=float, default=1.0, help="recon loss on pixel (x)")
    parser.add_argument("--lambda_fake_d", type=float, default=1.0)
    parser.add_argument("--lambda_rec_d", type=float, default=1.0)
    parser.add_argument("--lambda_fake_g", type=float, default=1.0)
    parser.add_argument("--lambda_rec_g", type=float, default=0)
    parser.add_argument("--pix_loss", type=str, default='l2')
    parser.add_argument("--no_train_g", action='store_true', help="do not update generator with encoder")
    parser.add_argument("--inception", type=str, default=None, help="path to precomputed inception embedding")
    parser.add_argument("--eval_every", type=int, default=1000, help="interval of metric evaluation")
    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument("--n_sample_fid", type=int, default=50000, help="number of the samples for calculating FID")
    parser.add_argument("--use_balanced_d_loss", action='store_true', help="adjust discriminator loss weights?")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--debug", type=str, default='none')
    parser.add_argument("--use_latent_d", action='store_true')

    args = parser.parse_args()
    util.seed_everything()
    args.device = device

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.n_mlp = 8
    args.n_latent = int(np.log2(args.size)) * 2 - 2
    args.latent = 512
    if args.which_latent == 'w_plus':
        args.latent_full = args.latent * args.n_latent
    elif args.which_latent == 'w_tied':
        args.latent_full = args.latent
    else:
        raise NotImplementedError

    args.start_iter = 0
    util.set_log_dir(args)
    util.print_args(parser, args)

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    from model import LatentDiscriminator
    discriminator = LatentDiscriminator(args.latent_full, 4).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) if args.g_reg_every > 0 else 1.
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1) if args.d_reg_every > 0 else 1.

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # Define Prior
    from model import LatentPrior
    prior = LatentPrior(args.latent, args.latent_full).to(device)
    p_optim = optim.Adam(
        prior.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    # Define Encoder
    reshape_latent = False
    if args.which_encoder == 'idinvert':
        from idinvert_pytorch.models.stylegan_encoder_network import StyleGANEncoderNet
        encoder = StyleGANEncoderNet(resolution=args.size, w_space_dim=args.latent,
            which_latent=args.which_latent, reshape_latent=reshape_latent,
            use_wscale=args.use_wscale).to(device)
        e_ema = StyleGANEncoderNet(resolution=args.size, w_space_dim=args.latent,
            which_latent=args.which_latent, reshape_latent=reshape_latent,
            use_wscale=args.use_wscale).to(device)
    else:
        from model import Encoder
        encoder = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
            which_latent=args.which_latent, reshape_latent=reshape_latent, stddev_group=args.stddev_group).to(device)
        e_ema = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
            which_latent=args.which_latent, reshape_latent=reshape_latent, stddev_group=args.stddev_group).to(device)
    e_ema.eval()
    accumulate(e_ema, encoder, 0)
    
    e_reg_ratio = 1.
    e_optim = optim.Adam(
        encoder.parameters(),
        lr=args.lr * e_reg_ratio,
        betas=(0 ** e_reg_ratio, 0.99 ** e_reg_ratio),
    )

    from idinvert_pytorch.models.perceptual_model import VGG16
    vggnet = VGG16(output_layer_idx=args.output_layer_idx).to(device)
    vgg_ckpt = torch.load(args.vgg_ckpt, map_location=lambda storage, loc: storage)
    vggnet.load_state_dict(vgg_ckpt)

    if args.resume and args.ckpt is None:
        args.ckpt = os.path.join(args.log_dir, 'weight', f"latest.pt")
    if args.ckpt is not None:  # resume
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            if 'iter' in ckpt:
                args.start_iter = ckpt["iter"]
            else:
                args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        encoder.load_state_dict(ckpt["e"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        e_ema.load_state_dict(ckpt["e_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        e_optim.load_state_dict(ckpt["e_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        encoder = nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        prior = nn.parallel.DistributedDataParallel(
            prior,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    dataset = None
    if args.dataset == 'multires':
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        dataset = MultiResolutionDataset(args.path, transform, args.size)
    elif args.dataset == 'videofolder':
        # [Note] Potentially, same transforms will be applied to a batch of images,
        # either a sequence or a pair (optical flow), so we should apply ToTensor first.
        transform = transforms.Compose(
            [
                # transforms.ToTensor(),  # this should be done in loader
                transforms.RandomHorizontalFlip(),
                transforms.Resize(args.size),  # Image.LANCZOS
                transforms.CenterCrop(args.size),
                # transforms.ToTensor(),  # normally placed here
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        dataset = VideoFolderDataset(args.path, transform, mode='image', cache=args.cache)
        if len(dataset) == 0:
            raise ValueError
    elif args.dataset == 'imagefolder':
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(args.size, Image.LANCZOS),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        dataset = datasets.ImageFolder(args.path, transform=transform)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=args.name)

    train(args, loader, generator, encoder, discriminator, prior, vggnet, g_optim, e_optim, d_optim, p_optim, g_ema, e_ema, device)
