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
from calc_inception import load_patched_inception_v3
from fid import extract_feature_from_samples, calc_fid, extract_feature_from_recon_hybrid
import pickle
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
    if model is not None:
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
        imgs = next(data_iter)
        samples.append(imgs)
        num -= imgs.size(0)
    samples = torch.cat(samples, dim=0)
    if num < 0:
        samples = samples[:num, ...]
    return samples


def load_real_samples(args, data_iter):
    npy_path = args.sample_cache
    # load sample_x
    if npy_path is not None and os.path.exists(npy_path):
        sample_x = torch.from_numpy(np.load(npy_path)).to(args.device)
    else:
        sample_x = accumulate_batches(data_iter, args.n_sample).to(args.device)
        if npy_path is not None:
            np.save(npy_path, sample_x.cpu().numpy())
    # load indice
    idx_path = None if npy_path is None else npy_path.replace('.npy', '_idx.npy')
    if idx_path is not None and os.path.exists(idx_path):
        sample_idx = torch.from_numpy(np.load(idx_path))
    else:
        sample_idx = torch.randperm(args.n_sample)
        if idx_path is not None:
            np.save(idx_path, sample_idx.numpy())
    return sample_x, sample_idx


def cross_reconstruction(encoder, generator, frames1, frames2, shuffle=True):
    batch = frames1.shape[0]
    if shuffle:
        # real: [frame1, frame2]
        # fake: [recon1, cross2]
        w1, _ = encoder(frames1)
        w2, _ = encoder(frames2)
        delta_w = w2 - w1
        delta_w = delta_w[torch.randperm(batch),...]
        x_recon, _ = generator([w1], input_is_latent=True, return_latents=False)
        x_real = frames1
        x_cross, _ = generator([w1 + delta_w], input_is_latent=True, return_latents=False)
        fake_img = torch.cat((x_recon, x_cross), 0)
        real_img = torch.cat((frames1, frames2), 0)
    else:
        # real: [frame11, frame21]
        # fake: [recon11, cross12]
        real_img11, real_img21 = frames1.chunk(2, dim=0)
        _, real_img22 = frames2.chunk(2, dim=0)
        w11, _ = encoder(real_img11)
        w21, _ = encoder(real_img21)
        w22, _ = encoder(real_img22)
        x_recon, _ = generator([w11], input_is_latent=True, return_latents=False)
        x_real = real_img11
        x_cross, _ = generator([w11 + w22 - w21], input_is_latent=True, return_latents=False)
        fake_img = torch.cat((x_recon, x_cross), 0)
        real_img = frames1
    return real_img, fake_img, x_real, x_recon, x_cross


def get_batch(loader, device, T=0, rand=True):
    frames = next(loader)  # [N, T, C, H, W]
    batch = frames.shape[0]
    if T <= 0:
        T = frames.shape[1]
    frames1 = frames[:,0,...]
    if rand:
        frames2 = frames[range(batch),torch.randint(1,T,(batch,)),...]
    else:
        frames2 = frames[:,T-1,...]
    frames1 = frames1.to(device)
    frames2 = frames2.to(device)
    return frames1, frames2


def train(args, loader, loader2, T_list,
          encoder, generator, discriminator, discriminator2, discriminator_w,
          vggnet, pwcnet, e_optim, g_optim, g1_optim, d_optim, d2_optim, dw_optim,
          e_ema, g_ema, device):
    inception = real_mean = real_cov = mean_latent = None
    if args.eval_every > 0:
        inception = nn.DataParallel(load_patched_inception_v3()).to(device)
        inception.eval()
        with open(args.inception, "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]
    if get_rank() == 0:
        if args.eval_every > 0:
            with open(os.path.join(args.log_dir, 'log_fid.txt'), 'a+') as f:
                f.write(f"Name: {getattr(args, 'name', 'NA')}\n{'-'*50}\n")
        if args.log_every > 0:
            with open(os.path.join(args.log_dir, 'log.txt'), 'a+') as f:
                f.write(f"Name: {getattr(args, 'name', 'NA')}\n{'-'*50}\n")

    loader = sample_data(loader)
    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0
    d_loss_val = 0
    e_loss_val = 0
    rec_loss_val = 0
    vgg_loss_val = 0
    adv_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    loss_dict = {"d": torch.tensor(0., device=device),
                 "real_score": torch.tensor(0., device=device),
                 "fake_score": torch.tensor(0., device=device),
                 "hybrid_score": torch.tensor(0., device=device),
                 "r1_d": torch.tensor(0., device=device),
                 "rec": torch.tensor(0., device=device),}
    avg_pix_loss = util.AverageMeter()
    avg_vgg_loss = util.AverageMeter()

    if args.distributed:
        e_module = encoder.module
        d_module = discriminator.module
        g_module = generator.module
    else:
        e_module = encoder
        d_module = discriminator
        g_module = generator
    d2_module = None
    if discriminator2 is not None:
        if args.distributed:
            d2_module = discriminator2.module
        else:
            d2_module = discriminator2

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 256, device)

    # sample_x = accumulate_batches(loader, args.n_sample).to(device)
    sample_x, sample_idx = load_real_samples(args, loader)
    assert (sample_x.shape[1] >= args.nframe_num)
    sample_x1 = sample_x[:,0,...]
    sample_x2 = sample_x[:,-1,...]
    fid_batch_idx = sample_idx = None  # Swap
    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    args.toggle_grads = True

    for idx in pbar:
        i = idx + args.start_iter

        if get_rank() == 0:
            if i % args.log_every == 0:
                with torch.no_grad():
                    e_eval = e_ema
                    e_eval.eval()
                    g_ema.eval()
                    nrow = int(args.n_sample ** 0.5)
                    nchw = list(sample_x1.shape)[1:]
                    # Recon
                    latent_real, _ = e_eval(sample_x1)
                    fake_img, _ = g_ema([latent_real], input_is_latent=True, return_latents=False)
                    sample = torch.cat((sample_x1.reshape(args.n_sample//nrow, nrow, *nchw), 
                                        fake_img.reshape(args.n_sample//nrow, nrow, *nchw)), 1)
                    utils.save_image(
                        sample.reshape(2*args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-recon.png"),
                        nrow=nrow,
                        normalize=True,
                        range=(-1, 1),
                    )
                    # Cross
                    w1, _ = e_eval(sample_x1)
                    w2, _ = e_eval(sample_x2)
                    dw = w2 - w1
                    dw = torch.cat(dw.chunk(2, 0)[::-1], 0) if sample_idx is None else dw[sample_idx,...]
                    fake_img, _ = g_ema([w1 + dw], input_is_latent=True, return_latents=False)
                    # sample = torch.cat((sample_x2.reshape(args.n_sample//nrow, nrow, *nchw), 
                    #                     fake_img.reshape(args.n_sample//nrow, nrow, *nchw)), 1)
                    drive = torch.cat((
                        torch.cat(sample_x1.chunk(2, 0)[::-1], 0).reshape(args.n_sample, 1, *nchw),
                        torch.cat(sample_x2.chunk(2, 0)[::-1], 0).reshape(args.n_sample, 1, *nchw),
                    ), 1)  # [n_sample, 2, C, H, w]
                    source = torch.cat((
                        sample_x1.reshape(args.n_sample, 1, *nchw),
                        fake_img.reshape(args.n_sample, 1, *nchw),
                    ), 1)  # [n_sample, 2, C, H, w]
                    sample = torch.cat((
                        drive.reshape(args.n_sample//nrow, 2*nrow, *nchw),
                        source.reshape(args.n_sample//nrow, 2*nrow, *nchw),
                    ), 1)  # [n_sample//nrow, 4*nrow, C, H, W]
                    utils.save_image(
                        sample.reshape(4*args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-cross.png"),
                        nrow=2*nrow,
                        normalize=True,
                        range=(-1, 1),
                    )
                    # Sample
                    sample, _ = g_ema([sample_z])
                    utils.save_image(
                        sample,
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-sample.png"),
                        nrow=nrow,
                        normalize=True,
                        range=(-1, 1),
                    )
                    # Fake hybrid samples
                    w1, _ = e_eval(sample_x1)
                    w2, _ = e_eval(sample_x2)
                    dw = w2 - w1
                    style_z = g_ema.get_styles([sample_z]).view(args.n_sample, -1)
                    if dw.shape[1] < style_z.shape[1]:  # W space
                        dw = dw.repeat(1, args.n_latent)
                    fake_img, _ = g_ema([style_z + dw], input_is_latent=True)
                    drive = torch.cat((
                        sample_x1.reshape(args.n_sample, 1, *nchw),
                        sample_x2.reshape(args.n_sample, 1, *nchw),
                    ), 1)
                    source = torch.cat((
                        sample.reshape(args.n_sample, 1, *nchw),
                        fake_img.reshape(args.n_sample, 1, *nchw),
                    ), 1)
                    sample = torch.cat((
                        drive.reshape(args.n_sample//nrow, 2*nrow, *nchw),
                        source.reshape(args.n_sample//nrow, 2*nrow, *nchw),
                    ), 1)
                    utils.save_image(
                        sample.reshape(4*args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-sample_hybrid.png"),
                        nrow=2*nrow,
                        normalize=True,
                        range=(-1, 1),
                    )
                e_eval.train()

        if i > args.iter:
            print("Done!")
            break

        # frames = next(loader)  # [N, T, C, H, W]
        # batch = frames.shape[0]
        # frames1 = frames[:,0,...]
        # frames2 = frames[range(batch),torch.randint(1,args.nframe_num,(batch,)),...]
        # frames1 = frames1.to(device)
        # frames2 = frames2.to(device)
        frames1, frames2 = get_batch(loader, device, T_list[i], not args.no_rand_T)
        batch = frames1.shape[0]

        # Train Discriminator
        requires_grad(encoder, False)
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(discriminator2, True)
        requires_grad(discriminator_w, True)

        real_img, fake_img, _, _, _ = cross_reconstruction(encoder, generator, frames1, frames2, args.shuffle)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img_aug, _ = augment(fake_img, ada_aug_p)
        else:
            real_img_aug = real_img
            fake_img_aug = fake_img
        
        fake_pred = discriminator(fake_img_aug)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        d_loss_gan = 0.
        if not args.decouple_d and args.lambda_gan_d > 0:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img, _ = generator(noise)
            if args.augment:
                fake_img, _ = augment(fake_img, ada_aug_p)
            fake_pred = discriminator(fake_img)
            fake_loss = F.softplus(fake_pred)
            d_loss_gan = fake_loss.mean()

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        fake_pred1, fake_pred2 = fake_pred.chunk(2, dim=0)
        loss_dict["fake_score"] = fake_pred1.mean()
        loss_dict["hybrid_score"] = fake_pred2.mean()

        discriminator.zero_grad()
        (d_loss + d_loss_gan * args.lambda_gan_d).backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat
        
        d_regularize = args.d_reg_every > 0 and i % args.d_reg_every == 0
        if d_regularize:
            # why not regularize on augmented real?
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss_d = d_r1_loss(real_pred, real_img)

            d_optim.zero_grad()
            (args.r1 / 2 * r1_loss_d * args.d_reg_every + 0 * real_pred.view(-1)[0]).backward()
            # Why 0* ? Answer is here https://github.com/rosinality/stylegan2-pytorch/issues/76
            d_optim.step()

            loss_dict["r1_d"] = r1_loss_d

        # Train Discriminator2
        if args.decouple_d and discriminator is not None:
            requires_grad(encoder, False)
            requires_grad(generator, False)
            requires_grad(discriminator2, True)
            real_img2 = real_img[:args.batch,...].detach()
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img2, _ = generator(noise)
            if args.augment:
                real_img2, _ = augment(real_img2, ada_aug_p)
                fake_img2, _ = augment(fake_img2, ada_aug_p)
            real_pred = discriminator2(real_img2)
            fake_pred = discriminator2(fake_img2)
            d2_loss = d_logistic_loss(real_pred, fake_pred)
            
            d_loss_fake_cross = 0.
            if args.lambda_fake_cross_d > 0:
                w1, _ = encoder(frames1)
                w2, _ = encoder(frames2)
                dw = w2 - w1
                noise = mixing_noise(args.batch, args.latent, args.mixing, device)
                style = generator.get_styles(noise).view(args.batch, -1)
                if dw.shape[1] < style.shape[1]:  # W space
                    dw = dw.repeat(1, args.n_latent)
                cross_img, _ = generator([style + dw], input_is_latent=True)
                fake_cross_pred = discriminator2(cross_img)
                d_loss_fake_cross = F.softplus(fake_cross_pred).mean()

            discriminator2.zero_grad()
            (d2_loss + d_loss_fake_cross * args.lambda_fake_cross_d).backward()
            d2_optim.step()
            d_regularize = args.d_reg_every > 0 and i % args.d_reg_every == 0
            if d_regularize:
                real_img2.requires_grad = True
                real_pred = discriminator2(real_img2)
                r1_loss = d_r1_loss(real_pred, real_img2)
                discriminator2.zero_grad()
                (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
                d2_optim.step()

        # Train Discriminator_W
        if args.learned_prior and args.lambda_gan_w > 0:
            requires_grad(encoder, False)
            requires_grad(generator, False)
            requires_grad(discriminator_w, True)
            # Real: encoded real images; Fake: style(noise)
            noise = mixing_noise(args.batch, args.latent, 0, device)
            fake_w = generator.get_latent(noise[0])
            real_w, _ = encoder(frames1)
            if fake_w.shape[1] < real_w.shape[1]:  # encodes in W+ space
                inject_index = random.randint(1, args.n_latent - 1)
                real_w = real_w.view(-1, args.n_latent, args.latent)[:,inject_index,:]
            fake_pred = discriminator_w(fake_w)
            real_pred = discriminator_w(real_w)
            dw_loss = d_logistic_loss(real_pred, fake_pred)
            dw_optim.zero_grad()
            dw_loss.backward()
            dw_optim.step()
            d_regularize = args.d_reg_every > 0 and i % args.d_reg_every == 0
            if d_regularize:
                real_w = encoder(frames1)[0].detach()
                real_w.requires_grad = True
                real_pred = discriminator_w(real_w)
                r1_loss_dw = d_r1_loss(real_pred, real_w)
                dw_optim.zero_grad()
                (args.r1 / 2 * r1_loss_dw * args.d_reg_every + 0 * real_pred.view(-1)[0]).backward()
                dw_optim.step()

        # Train Encoder and Generator
        requires_grad(encoder, True)
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(discriminator2, False)
        requires_grad(discriminator_w, False)
        pix_loss = vgg_loss = adv_loss = rec_loss = torch.tensor(0., device=device)

        _, fake_img, x_real, x_recon, x_cross = cross_reconstruction(encoder, generator, frames1, frames2, args.shuffle)

        if args.lambda_adv > 0:
            if args.augment:
                fake_img_aug, _ = augment(fake_img, ada_aug_p)
            else:
                fake_img_aug = fake_img
            fake_pred = discriminator(fake_img_aug)
            adv_loss = g_nonsaturating_loss(fake_pred)

        if args.lambda_pix > 0:
            if args.pix_loss == 'l2':
                pix_loss = torch.mean((x_recon - x_real) ** 2)
            else:
                pix_loss = F.l1_loss(x_recon, x_real)

        if args.lambda_vgg > 0:
            real_feat = vggnet(x_real)
            fake_feat = vggnet(x_recon)
            vgg_loss = torch.mean((fake_feat - real_feat) ** 2)

        e_loss = (adv_loss * args.lambda_adv + 
            pix_loss * args.lambda_pix + vgg_loss * args.lambda_vgg)
        loss_dict["e"] = e_loss
        loss_dict["pix"] = pix_loss
        loss_dict["vgg"] = vgg_loss
        loss_dict["adv"] = adv_loss

        e_optim.zero_grad()
        g_optim.zero_grad()
        e_loss.backward()
        e_optim.step()
        g_optim.step()
        
        if args.lambda_gan_g > 0:
            # Train Generator
            requires_grad(encoder, False)
            requires_grad(generator, True)
            requires_grad(discriminator, False)
            requires_grad(discriminator2, False)
            requires_grad(discriminator_w, False)

            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img, _ = generator(noise)
            if args.augment:
                fake_img, _ = augment(fake_img, ada_aug_p)
            if not args.decouple_d:
                fake_pred = discriminator(fake_img)
            else:
                fake_pred = discriminator2(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred) * args.lambda_gan_g
            generator.zero_grad()
            g_loss.backward()
            generator.style.zero_grad()
            g_optim.step()

            g_loss_fake_cross = 0.
            if args.lambda_fake_cross_g > 0:
                w1, _ = encoder(frames1)
                w2, _ = encoder(frames2)
                dw = w2 - w1
                noise = mixing_noise(args.batch, args.latent, args.mixing, device)
                style = generator.get_styles(noise).view(args.batch, -1)
                if dw.shape[1] < style.shape[1]:  # W space
                    dw = dw.repeat(1, args.n_latent)
                cross_img, _ = generator([style + dw], input_is_latent=True)
                fake_cross_pred = discriminator2(cross_img)
                g_loss_fake_cross = g_nonsaturating_loss(fake_cross_pred)
                generator.zero_grad()
                (g_loss_fake_cross * args.lambda_fake_cross_g).backward()
                g_optim.step()

            g_regularize = args.g_reg_every > 0 and i % args.g_reg_every == 0
            if g_regularize:
                path_batch_size = max(1, args.batch // args.path_batch_shrink)
                noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
                fake_img, latents = generator(noise, return_latents=True)
                path_loss, mean_path_length, path_lengths = g_path_regularize(
                    fake_img, latents, mean_path_length
                )
                generator.zero_grad()
                weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss
                if args.path_batch_shrink:
                    weighted_path_loss += 0 * fake_img[0, 0, 0, 0]
                weighted_path_loss.backward()
                g_optim.step()
                # mean_path_length_avg = (
                #     reduce_sum(mean_path_length).item() / get_world_size()
                # )
            loss_dict["path"] = path_loss
            loss_dict["path_length"] = path_lengths.mean()

        if args.learned_prior:
            g1_loss = 0.
            if args.lambda_gan_w > 0:
                # Use latent discriminator to update g1
                noise = mixing_noise(args.batch, args.latent, 0, device)
                fake_w = generator.get_latent(noise[0])
                fake_pred = discriminator_w(fake_w)
                g1_loss += g_nonsaturating_loss(fake_pred) * args.lambda_gan_w
            if args.lambda_adv_w > 0:
                # Use image discriminator to update g1
                noise = mixing_noise(args.batch, args.latent, args.mixing, device)
                fake_img, _ = generator(noise)
                fake_pred = discriminator(fake_img)
                g1_loss += g_nonsaturating_loss(fake_pred) * args.lambda_adv_w
            if args.lambda_rec_w > 0:
                requires_grad(encoder, False)
                # G(g1(z)) should be recognized by E
                noise = mixing_noise(args.batch, args.latent, 0, device)
                fake_img, fake_w = generator(noise, return_latents=True)
                if args.which_latent == 'w_plus':
                    fake_w = fake_w.view(args.batch, -1)
                else:
                    fake_w = fake_w[:,0,:]
                w_pred, _ = encoder(fake_img)
                g1_loss += torch.mean((w_pred - fake_w) ** 2) * args.lambda_rec_w
            g1_optim.zero_grad()
            g1_loss.backward()
            g1_optim.step()

        accumulate(e_ema, e_module, accum)
        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        e_loss_val = loss_reduced["e"].mean().item()
        r1_d_val = loss_reduced["r1_d"].mean().item()
        pix_loss_val = loss_reduced["pix"].mean().item()
        vgg_loss_val = loss_reduced["vgg"].mean().item()
        adv_loss_val = loss_reduced["adv"].mean().item()
        rec_loss_val = loss_reduced["rec"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        hybrid_score_val = loss_reduced["hybrid_score"].mean().item()
        # path_loss_val = loss_reduced["path"].mean().item()
        # path_length_val = loss_reduced["path_length"].mean().item()
        avg_pix_loss.update(pix_loss_val, real_img.shape[0])
        avg_vgg_loss.update(vgg_loss_val, real_img.shape[0])

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; e: {e_loss_val:.4f}; r1_d: {r1_d_val:.4f}; "
                    f"pix: {pix_loss_val:.4f}; vgg: {vgg_loss_val:.4f}; adv: {adv_loss_val:.4f}; "
                    # f"path: {path_loss_val:.4f}; augment: {ada_aug_p:.4f}"
                )
            )

            if i % args.log_every == 0:
                with torch.no_grad():
                    latent_x, _ = e_ema(sample_x1)
                    fake_x, _ = generator([latent_x], input_is_latent=True, return_latents=False)
                    sample_pix_loss = torch.sum((sample_x1 - fake_x) ** 2)
                with open(os.path.join(args.log_dir, 'log.txt'), 'a+') as f:
                    f.write(f"{i:07d}; pix: {avg_pix_loss.avg}; vgg: {avg_vgg_loss.avg}; "
                            f"ref: {sample_pix_loss.item()};\n")
            
            if args.eval_every > 0 and i % args.eval_every == 0:
                with torch.no_grad():
                    g_ema.eval()
                    e_ema.eval()
                    # Sample FID
                    if args.truncation < 1:
                        mean_latent = g_ema.mean_latent(4096)
                    features = extract_feature_from_samples(
                        g_ema, inception, args.truncation, mean_latent, 64, args.n_sample_fid, args.device
                    ).numpy()
                    sample_mean = np.mean(features, 0)
                    sample_cov = np.cov(features, rowvar=False)
                    fid_sa = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
                    # Recon FID
                    features = extract_feature_from_recon_hybrid(
                        e_ema, g_ema, inception, args.truncation, mean_latent, loader2, args.device,
                        mode='recon',
                    ).numpy()
                    sample_mean = np.mean(features, 0)
                    sample_cov = np.cov(features, rowvar=False)
                    fid_re = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
                    # Hybrid FID
                    features = extract_feature_from_recon_hybrid(
                        e_ema, g_ema, inception, args.truncation, mean_latent, loader2, args.device,
                        mode='hybrid', # shuffle_idx=fid_batch_idx
                    ).numpy()
                    sample_mean = np.mean(features, 0)
                    sample_cov = np.cov(features, rowvar=False)
                    fid_hy = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
                # print("Sample FID:", fid_sa, "Recon FID:", fid_re, "Hybrid FID:", fid_hy)
                with open(os.path.join(args.log_dir, 'log_fid.txt'), 'a+') as f:
                    f.write(f"{i:07d}; sample fid: {float(fid_sa):.4f}; recon fid: {float(fid_re):.4f}; hybrid fid: {float(fid_hy):.4f};\n")

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Encoder": e_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1 D": r1_d_val,
                        "Pix Loss": pix_loss_val,
                        "VGG Loss": vgg_loss_val,
                        "Adv Loss": adv_loss_val,
                        "Rec Loss": rec_loss_val,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Hybrid Score": hybrid_score_val,
                    }
                )

            if i % args.save_every == 0:
                e_eval = e_ema
                torch.save(
                    {
                        "e": e_module.state_dict(),
                        "d": d_module.state_dict(),
                        "d2": d2_module.state_dict() if args.decouple_d else None,
                        "g": g_module.state_dict(),
                        "g_ema": g_module.state_dict(),
                        "e_ema": e_eval.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "d2_optim": d2_optim.state_dict() if args.decouple_d else None,
                        "g_optim": g_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                        "iter": i,
                    },
                    os.path.join(args.log_dir, 'weight', f"{str(i).zfill(6)}.pt"),
                )
            
            if i % args.save_latest_every == 0:
                torch.save(
                    {
                        "e": e_module.state_dict(),
                        "d": d_module.state_dict(),
                        "d2": d2_module.state_dict() if args.decouple_d else None,
                        "g": g_module.state_dict(),
                        "g_ema": g_module.state_dict(),
                        "e_ema": e_eval.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "d2_optim": d2_optim.state_dict() if args.decouple_d else None,
                        "g_optim": g_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                        "iter": i,
                    },
                    os.path.join(args.log_dir, 'weight', f"latest.pt"),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 encoder trainer")

    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--dataset", type=str, default='videofolder')
    parser.add_argument("--cache", type=str, default='local.db')
    parser.add_argument("--sample_cache", type=str, default=None)
    parser.add_argument("--name", type=str, help="experiment name", default='default_exp')
    parser.add_argument("--log_root", type=str, help="where to save training logs", default='logs')
    parser.add_argument("--log_every", type=int, default=100, help="save samples every # iters")
    parser.add_argument("--save_every", type=int, default=1000, help="save checkpoints every # iters")
    parser.add_argument("--save_latest_every", type=int, default=100, help="save latest checkpoints every # iters")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--toggle_grads", action='store_true')
    parser.add_argument("--use_optical_flow", action='store_true')
    parser.add_argument("--use_wscale", action='store_true', help="whether to use `wscale` layer in idinvert encoder")
    parser.add_argument("--no_ema", action='store_true', help="do not use ema if enabled")
    parser.add_argument("--train_on_fake", action='store_true', help="train encoder on fake?")
    parser.add_argument("--e_rec_every", type=int, default=1, help="interval of minimizing recon loss on w")
    parser.add_argument("--pix_loss", type=str, default='l2')
    parser.add_argument("--lambda_pix", type=float, default=1.0, help="recon loss on pixel (x)")
    parser.add_argument("--lambda_vgg", type=float, default=5e-5)
    parser.add_argument("--lambda_adv", type=float, default=0.1)
    parser.add_argument("--lambda_gan_d", type=float, default=0., help="train a gan branch?")
    parser.add_argument("--lambda_gan_g", type=float, default=0., help="train a gan branch?")
    parser.add_argument("--lambda_rec", type=float, default=1.0, help="recon loss on style (w)")
    parser.add_argument("--lambda_adv_w", type=float, default=0., help="adversarial loss from image discriminator")
    parser.add_argument("--lambda_gan_w", type=float, default=0., help="adversarial loss from latent discriminator")
    parser.add_argument("--lambda_mmd_w", type=float, default=0.)
    parser.add_argument("--lambda_rec_w", type=float, default=0.)
    parser.add_argument("--output_layer_idx", type=int, default=23)
    parser.add_argument("--vgg_ckpt", type=str, default="vgg16.pth")
    parser.add_argument("--which_encoder", type=str, default='style')
    parser.add_argument("--which_latent", type=str, default='w_shared')
    parser.add_argument("--stddev_group", type=int, default=4)
    parser.add_argument("--nframe_num", type=int, default=5)
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--learned_prior", action='store_true', help="learned latent prior (w)?")
    parser.add_argument("--train_from_scratch", action='store_true')
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
        default=16,
        help="interval of the applying r1 regularization, no if 0",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=0,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--e_reg_every",
        type=int,
        default=0,
        help="interval of the applying r1 regularization, no if 0",
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
        "--d_ckpt",
        type=str,
        default=None,
        help="path to the checkpoint of discriminator",
    )
    parser.add_argument(
        "--d2_ckpt",
        type=str,
        default=None,
        help="path to the checkpoint of discriminator2",
    )
    parser.add_argument(
        "--e_ckpt",
        type=str,
        default=None,
        help="path to the checkpoint of encoder",
    )
    parser.add_argument(
        "--g_ckpt",
        type=str,
        default=None,
        help="path to the checkpoint of generator",
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
    parser.add_argument("--inception", type=str, default=None, help="path to precomputed inception embedding")
    parser.add_argument("--eval_every", type=int, default=1000, help="interval of metric evaluation")
    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument("--n_sample_fid", type=int, default=10000, help="number of the samples for calculating FID")
    parser.add_argument("--decouple_d", action='store_true')
    parser.add_argument("--lambda_fake_cross_d", type=float, default=0)
    parser.add_argument("--lambda_fake_cross_g", type=float, default=0)
    parser.add_argument("--no_rand_T", action='store_true')
    parser.add_argument('--nframe_num_range', type=util.str2list, default=[])
    parser.add_argument('--nframe_iter_range', type=util.str2list, default=[])

    args = parser.parse_args()
    util.seed_everything(0)
    args.device = device

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
    elif args.which_latent == 'w_shared':
        args.latent_full = args.latent
    else:
        raise NotImplementedError
    args.n_mlp = 8
    args.use_latent_discriminator = args.learned_prior and args.lambda_gan_w > 0

    args.start_iter = 0
    util.set_log_dir(args)
    util.print_args(parser, args)
    
    # Auxiliary models (VGG and PWC)
    vggnet = VGG16(output_layer_idx=args.output_layer_idx).to(device)
    vgg_ckpt = torch.load(args.vgg_ckpt, map_location=lambda storage, loc: storage)
    vggnet.load_state_dict(vgg_ckpt)

    pwcnet = None
    # if args.use_optical_flow:
    #     pwc = __import__('pytorch-pwc.run', globals(), locals(), ['Network'], 0)
    #     pwcnet = pwc.Network().to(device)  # state_dict loaded in init
    #     pwcnet.eval()

    in_channel = 3
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, in_channel=in_channel,
    ).to(device)
    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    e_ema = None
    if args.which_encoder == 'idinvert':
        from idinvert_pytorch.models.stylegan_encoder_network import StyleGANEncoderNet
        encoder = StyleGANEncoderNet(resolution=args.size, w_space_dim=args.latent,
            which_latent=args.which_latent, reshape_latent=False,
            use_wscale=args.use_wscale).to(device)
        e_ema = StyleGANEncoderNet(resolution=args.size, w_space_dim=args.latent,
            which_latent=args.which_latent, reshape_latent=False,
            use_wscale=args.use_wscale).to(device)
    else:
        from model import Encoder
        encoder = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
            which_latent=args.which_latent, reshape_latent=False, stddev_group=args.stddev_group).to(device)
        e_ema = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
            which_latent=args.which_latent, reshape_latent=False, stddev_group=args.stddev_group).to(device)
    e_ema.eval()
    accumulate(e_ema, encoder, 0)

    # For lazy regularization (see paper appendix page 11)
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) if args.g_reg_every > 0 else 1.
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1) if args.d_reg_every > 0 else 1.
    e_reg_ratio = 1.
    
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    g1_optim = optim.Adam(  # rmsprop, sgd w mom
        generator.style.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    e_optim = optim.Adam(
        encoder.parameters(),
        lr=args.lr * e_reg_ratio,
        betas=(0 ** e_reg_ratio, 0.99 ** e_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    discriminator_w = dw_optim = None
    if args.use_latent_discriminator:
        from model import LatentDiscriminator
        discriminator_w = LatentDiscriminator(args.latent, 4).to(device)
        dw_optim = optim.Adam(
            discriminator_w.parameters(),
            lr=args.lr * 1,
            betas=(0 ** 1, 0.99 ** 1),
        )
    
    discriminator2 = d2_optim = None
    if args.decouple_d:
        discriminator2 = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier,
        ).to(device)
        d2_optim = optim.Adam(
            discriminator2.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

    if args.resume:
        if args.ckpt is None:
            args.ckpt = os.path.join(args.log_dir, 'weight', f"latest.pt")
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
        encoder.load_state_dict(ckpt["e"])
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        e_ema.load_state_dict(ckpt["e_ema"])
        g_ema.load_state_dict(ckpt["g_ema"])
        e_optim.load_state_dict(ckpt["e_optim"])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])
        if discriminator2 is not None:
            discriminator2.load_state_dict(ckpt["d2"])
            d2_optim.load_state_dict(ckpt["d2_optim"])
    elif not args.train_from_scratch:
        # if e_ckpt is provided, load encoder as warm start, else train encoder from scratch
        # if g_ckpt is provided, load generator as warm start, else train generator from scratch
        if args.e_ckpt is not None:
            print("load e model:", args.e_ckpt)
            e_ckpt = torch.load(args.e_ckpt, map_location=lambda storage, loc: storage)
            encoder.load_state_dict(e_ckpt["e"])
            e_ema.load_state_dict(e_ckpt["e_ema"])
            e_optim.load_state_dict(e_ckpt["e_optim"])
        if args.g_ckpt is not None:
            print("load g model:", args.g_ckpt)
            g_ckpt = torch.load(args.g_ckpt, map_location=lambda storage, loc: storage)
            generator.load_state_dict(g_ckpt["g"])
            g_ema.load_state_dict(g_ckpt["g_ema"])
            g_optim.load_state_dict(g_ckpt["g_optim"])
        if args.d_ckpt is not None:
            print("load d model:", args.d_ckpt)
            d_ckpt = torch.load(args.d_ckpt, map_location=lambda storage, loc: storage)
            discriminator.load_state_dict(d_ckpt["d"])
            d_optim.load_state_dict(d_ckpt["d_optim"])
        if args.d2_ckpt is not None:
            # D2 is for gan branch, D2 is still named 'd' in d2_ckpt
            print("load d model:", args.d2_ckpt)
            d2_ckpt = torch.load(args.d2_ckpt, map_location=lambda storage, loc: storage)
            discriminator2.load_state_dict(d2_ckpt["d"])
            d2_optim.load_state_dict(d2_ckpt["d_optim"])

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
        if args.use_latent_discriminator:
            discriminator_w = nn.parallel.DistributedDataParallel(
                discriminator_w,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )
        if discriminator2 is not None:
            discriminator2 = nn.parallel.DistributedDataParallel(
                discriminator2,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )
    T_list = util.get_nframe_num(args)
    dataset = None
    if args.dataset == 'multires':
        # TODO: force G(w+Dy) to be real
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
        dataset = VideoFolderDataset(args.path, transform, cache=args.cache, unbind=False,
                                     mode='nframe', nframe_num=args.nframe_num)
        if len(dataset) == 0:
            raise ValueError
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    # A subset of length args.n_sample_fid for FID evaluation
    loader2 = None
    if args.eval_every > 0:
        indices = torch.randperm(len(dataset))[:args.n_sample_fid]
        dataset2 = data.Subset(dataset, indices)
        loader2 = data.DataLoader(dataset2, batch_size=64, num_workers=4, shuffle=False)

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=args.name)

    train(args, loader, loader2, T_list,
          encoder, generator, discriminator, discriminator2, discriminator_w,
          vggnet, pwcnet, e_optim, g_optim, g1_optim, d_optim, d2_optim, dw_optim,
          e_ema, g_ema, device)
