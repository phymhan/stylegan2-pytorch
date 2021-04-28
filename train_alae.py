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
from fid import extract_feature_from_samples, calc_fid, extract_feature_from_reconstruction
import pickle
import pdb
st = pdb.set_trace

try:
    import wandb

except ImportError:
    wandb = None

from dataset import get_image_dataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment


def scale_grad(model, scale):
    if model is not None:
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= scale


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
    # Endless image iterator
    while True:
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                yield batch[0]
            else:
                yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
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


def calculate_adaptive_weight(nll_loss, g_loss, last_layer=None):
    if last_layer is None:
        return 1.0
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)
    if isinstance(last_layer, (list, tuple)):
        nll_grads = torch.cat([v.view(-1) for v in nll_grads])
        g_grads = torch.cat([v.view(-1) for v in g_grads])
    else:
        nll_grads = nll_grads[0]
        g_grads = g_grads[0]
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight


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
        if isinstance(imgs, (list, tuple)):
            imgs = imgs[0]
        samples.append(imgs)
        num -= imgs.size(0)
    samples = torch.cat(samples, dim=0)
    if num < 0:
        samples = samples[:num, ...]
    return samples


def load_real_samples(args, data_iter):
    npy_path = args.sample_cache
    if npy_path is not None and os.path.exists(npy_path):
        sample_x = torch.from_numpy(np.load(npy_path)).to(args.device)
    else:
        sample_x = accumulate_batches(data_iter, args.n_sample).to(args.device)
        if npy_path is not None:
            np.save(npy_path, sample_x.cpu().numpy())
    return sample_x


def train(args, loader, loader2, generator, encoder, discriminator, discriminator2,
          vggnet, g_optim, e_optim, d_optim, d2_optim, g_ema, e_ema, device):
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
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {
        'recx_score': torch.tensor(0.0, device=device),
        'ae_fake': torch.tensor(0.0, device=device),
        'ae_real': torch.tensor(0.0, device=device),
        'pix': torch.tensor(0.0, device=device),
        'vgg': torch.tensor(0.0, device=device),
    }
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

    d2_module = None
    if discriminator2 is not None:
        if args.distributed:
            d2_module = discriminator2.module
        else:
            d2_module = discriminator2

    # When joint training enabled, d_weight balances reconstruction loss and adversarial loss on 
    # recontructed real images. This does not balance the overall AE loss and GAN loss.
    d_weight = torch.tensor(1.0, device=device)
    last_layer = None
    if args.use_adaptive_weight:
        if args.distributed:
            last_layer = generator.module.get_last_layer()
        else:
            last_layer = generator.get_last_layer()
    g_scale = 1

    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0
    r_t_dict = {'real': 0, 'fake': 0, 'recx': 0}  # r_t stat

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, args.ada_every, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    sample_x = load_real_samples(args, loader)
    if sample_x.ndim > 4:
        sample_x = sample_x[:,0,...]
    
    input_is_latent = args.latent_space != 'z'  # Encode in z space?

    n_step_max = max(args.n_step_d, args.n_step_e)

    requires_grad(g_ema, False)
    requires_grad(e_ema, False)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        if args.debug: util.seed_everything(i)
        real_imgs = [next(loader).to(device) for _ in range(n_step_max)]

        # Train Discriminator and Encoder
        requires_grad(generator, False)
        requires_grad(encoder, True)
        requires_grad(discriminator, True)
        requires_grad(discriminator2, True)
        for step_index in range(args.n_step_d):
            real_img = real_imgs[step_index]
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img, _ = generator(noise)
            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)
                fake_img_aug, _ = augment(fake_img, ada_aug_p)
            else:
                real_img_aug = real_img
                fake_img_aug = fake_img
            real_pred = discriminator(encoder(real_img_aug)[0])
            fake_pred = discriminator(encoder(fake_img_aug)[0])
            d_loss_real = F.softplus(-real_pred).mean()
            d_loss_fake = F.softplus(fake_pred).mean()
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()

            d_loss_rec = 0.
            if args.lambda_rec_d > 0:
                latent_real, _ = encoder(real_img)
                rec_img, _ = generator([latent_real], input_is_latent=input_is_latent)
                if args.augment:
                    rec_img, _ = augment(rec_img, ada_aug_p)
                rec_pred = discriminator(encoder(rec_img)[0])
                d_loss_rec = F.softplus(rec_pred).mean()
                loss_dict["recx_score"] = rec_pred.mean()
                r_t_dict['recx'] = torch.sign(rec_pred).sum().item() / args.batch

            d_loss = d_loss_real + d_loss_fake * args.lambda_fake_d + d_loss_rec * args.lambda_rec_d
            loss_dict["d"] = d_loss

            discriminator.zero_grad()
            encoder.zero_grad()
            d_loss.backward()
            d_optim.step()
            e_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat
        r_t_dict['real'] = torch.sign(real_pred).sum().item() / args.batch
        r_t_dict['fake'] = torch.sign(fake_pred).sum().item() / args.batch

        d_regularize = args.d_reg_every > 0 and i % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True
            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)
            else:
                real_img_aug = real_img
            real_pred = discriminator(encoder(real_img_aug)[0])
            r1_loss = d_r1_loss(real_pred, real_img)
            discriminator.zero_grad()
            encoder.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()
            e_optim.step()
        loss_dict["r1"] = r1_loss

        # Train Generator
        requires_grad(generator, True)
        requires_grad(encoder, False)
        requires_grad(discriminator, False)
        requires_grad(discriminator2, False)
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)
        if args.augment:
            fake_img_aug, _ = augment(fake_img, ada_aug_p)
        else:
            fake_img_aug = fake_img
        fake_pred = discriminator(encoder(fake_img_aug)[0])
        g_loss_fake = g_nonsaturating_loss(fake_pred)
        loss_dict["g"] = g_loss_fake
        generator.zero_grad()
        (g_loss_fake * args.lambda_fake_g).backward()
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
            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )
        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        # Train Encoder (and Generator)
        joint = (not args.no_joint) and (g_scale > 1e-6)

        # Train AE on fake samples (latent reconstruction)
        if args.lambda_rec_w > 0:
            requires_grad(encoder, True)
            requires_grad(generator, joint)
            requires_grad(discriminator, False)
            requires_grad(discriminator2, False)
            for step_index in range(args.n_step_e):
                mixing_prob = 0 if args.which_latent == 'w_tied' else args.mixing
                noise = mixing_noise(args.batch, args.latent, mixing_prob, device)
                fake_img, latent_fake = generator(noise, return_latents=True, detach_style=not args.no_detach_style)
                if args.which_latent == 'w_tied':
                    latent_fake = latent_fake[:,0,:]
                else:
                    latent_fake = latent_fake.view(args.batch, -1)
                latent_pred, _ = encoder(fake_img)
                ae_loss_fake = torch.mean((latent_pred - latent_fake.detach()) ** 2)
                loss_dict["ae_fake"] = ae_loss_fake

                if joint:
                    encoder.zero_grad()
                    generator.zero_grad()
                    (ae_loss_fake * args.lambda_rec_w).backward()
                    e_optim.step()
                    if args.g_decay is not None:
                        scale_grad(generator, g_scale)
                    # Do NOT update F (or generator.style). Grad should be zero when style 
                    # is detached in generator, but we explicitly zero it, just in case.
                    if not args.no_detach_style:
                        generator.style.zero_grad()
                    g_optim.step()
                else:
                    encoder.zero_grad()
                    (ae_loss_fake * args.lambda_rec_w).backward()
                    e_optim.step()

        # Train AE on real samples (image reconstruction)
        if args.lambda_pix + args.lambda_vgg + args.lambda_adv > 0:
            requires_grad(encoder, True)
            requires_grad(generator, joint)
            requires_grad(discriminator, False)
            requires_grad(discriminator2, False)
            pix_loss = vgg_loss = adv_loss = torch.tensor(0., device=device)
            for step_index in range(args.n_step_e):
                real_img = real_imgs[step_index]
                latent_real, _ = encoder(real_img)
                rec_img, _ = generator([latent_real], input_is_latent=input_is_latent)
                if args.lambda_pix > 0:
                    if args.pix_loss == 'l2':
                        pix_loss = torch.mean((rec_img - real_img) ** 2)
                    elif args.pix_loss == 'l1':
                        pix_loss = F.l1_loss(rec_img, real_img)
                if args.lambda_vgg > 0:
                    vgg_loss = torch.mean((vggnet(real_img) - vggnet(rec_img)) ** 2)
                if args.lambda_adv > 0:
                    if args.augment:
                        rec_img_aug, _ = augment(rec_img, ada_aug_p)
                    else:
                        rec_img_aug = rec_img
                    rec_pred = discriminator(encoder(rec_img_aug)[0])
                    adv_loss = g_nonsaturating_loss(rec_pred)

                if args.use_adaptive_weight and i >= args.disc_iter_start:
                    nll_loss = pix_loss * args.lambda_pix + vgg_loss * args.lambda_vgg
                    g_loss = adv_loss * args.lambda_adv
                    d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)

                ae_loss_real = (
                    pix_loss * args.lambda_pix + vgg_loss * args.lambda_vgg + 
                    d_weight * adv_loss * args.lambda_adv
                )
                loss_dict["ae_real"] = ae_loss_real
                loss_dict["pix"] = pix_loss
                loss_dict["vgg"] = vgg_loss
                loss_dict["adv"] = adv_loss

                if joint:
                    encoder.zero_grad()
                    generator.zero_grad()
                    ae_loss_real.backward()
                    e_optim.step()
                    if args.g_decay is not None:
                        scale_grad(generator, g_scale)
                    g_optim.step()
                else:
                    encoder.zero_grad()
                    ae_loss_real.backward()
                    e_optim.step()

        if args.g_decay is not None:
            g_scale *= args.g_decay

        # Update EMA
        ema_nimg = args.ema_kimg * 1000
        if args.ema_rampup is not None:
            ema_nimg = min(ema_nimg, i * args.batch * args.ema_rampup)
        accum = 0.5 ** (args.batch / max(ema_nimg, 1e-8))
        accumulate(g_ema, g_module, 0 if args.no_ema_g else accum)
        accumulate(e_ema, e_module, 0 if args.no_ema_e else accum)

        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        ae_real_val = loss_reduced["ae_real"].mean().item()
        ae_fake_val = loss_reduced["ae_fake"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        recx_score_val = loss_reduced["recx_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()
        pix_loss_val = loss_reduced["pix"].mean().item()
        vgg_loss_val = loss_reduced["vgg"].mean().item()
        avg_pix_loss.update(pix_loss_val, real_img.shape[0])
        avg_vgg_loss.update(vgg_loss_val, real_img.shape[0])

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"ae_fake: {ae_fake_val:.4f}; ae_real: {ae_real_val:.4f}; "
                    f"g: {g_loss_val:.4f}; path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}; "
                    f"d_weight: {d_weight.item():.4f}; "
                )
            )

            if i % args.log_every == 0:
                with torch.no_grad():
                    g_ema.eval()
                    e_ema.eval()
                    nrow = int(args.n_sample ** 0.5)
                    nchw = list(sample_x.shape)[1:]
                    # Reconstruction of real images
                    latent_x, _ = e_ema(sample_x)
                    rec_real, _ = g_ema([latent_x], input_is_latent=input_is_latent)
                    sample = torch.cat((sample_x.reshape(args.n_sample//nrow, nrow, *nchw), 
                                        rec_real.reshape(args.n_sample//nrow, nrow, *nchw)), 1)
                    utils.save_image(
                        sample.reshape(2*args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-recon.png"),
                        nrow=nrow,
                        normalize=True,
                        range=(-1, 1),
                    )
                    ref_pix_loss = torch.sum(torch.abs(sample_x - rec_real))
                    ref_vgg_loss = torch.mean((vggnet(sample_x) - vggnet(rec_real)) ** 2) if vggnet is not None else 0
                    # Fixed fake samples and reconstructions
                    sample_gz, _ = g_ema([sample_z])
                    latent_gz, _ = e_ema(sample_gz)
                    rec_fake, _ = g_ema([latent_gz], input_is_latent=input_is_latent)
                    sample = torch.cat((sample_gz.reshape(args.n_sample//nrow, nrow, *nchw), 
                                        rec_fake.reshape(args.n_sample//nrow, nrow, *nchw)), 1)
                    utils.save_image(
                        sample.reshape(2*args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-sample.png"),
                        nrow=nrow,
                        normalize=True,
                        range=(-1, 1),
                    )

                with open(os.path.join(args.log_dir, 'log.txt'), 'a+') as f:
                    f.write(
                        (
                            f"{i:07d}; "
                            f"d: {d_loss_val:.4f}; r1: {r1_val:.4f}; "
                            f"ae_fake: {ae_fake_val:.4f}; ae_real: {ae_real_val:.4f}; "
                            f"g: {g_loss_val:.4f}; path: {path_loss_val:.4f}; mean_path: {mean_path_length_avg:.4f}; "
                            f"augment: {ada_aug_p:.4f}; {'; '.join([f'{k}: {r_t_dict[k]:.4f}' for k in r_t_dict])}; "
                            f"real_score: {real_score_val:.4f}; fake_score: {fake_score_val:.4f}; recx_score: {recx_score_val:.4f}; "
                            f"pix: {avg_pix_loss.avg:.4f}; vgg: {avg_vgg_loss.avg:.4f}; "
                            f"ref_pix: {ref_pix_loss.item():.4f}; ref_vgg: {ref_vgg_loss.item():.4f}; "
                            f"d_weight: {d_weight.item():.4f}; "
                            f"\n"
                        )
                    )

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

            if args.eval_every > 0 and i % args.eval_every == 0:
                with torch.no_grad():
                    fid_sa = fid_re = fid_sr = 0
                    g_ema.eval()
                    e_ema.eval()
                    if args.truncation < 1:
                        mean_latent = g_ema.mean_latent(4096)
                    # Sample FID
                    if 'fid_sample' in args.which_metric:
                        features = extract_feature_from_samples(
                            g_ema, inception, args.truncation, mean_latent, 64, args.n_sample_fid, args.device
                        ).numpy()
                        sample_mean = np.mean(features, 0)
                        sample_cov = np.cov(features, rowvar=False)
                        fid_sa = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
                    # Sample reconstruction FID
                    if 'fid_sample_recon' in args.which_metric:
                        features = extract_feature_from_samples(
                            g_ema, inception, args.truncation, mean_latent, 64, args.n_sample_fid, args.device,
                            mode='recon', encoder=e_ema, input_is_latent=input_is_latent,
                        ).numpy()
                        sample_mean = np.mean(features, 0)
                        sample_cov = np.cov(features, rowvar=False)
                        fid_sr = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
                    # Real reconstruction FID
                    if 'fid_recon' in args.which_metric:
                        features = extract_feature_from_reconstruction(
                            e_ema, g_ema, inception, args.truncation, mean_latent, loader2, args.device,
                            input_is_latent=input_is_latent, mode='recon',
                        ).numpy()
                        sample_mean = np.mean(features, 0)
                        sample_cov = np.cov(features, rowvar=False)
                        fid_re = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
                with open(os.path.join(args.log_dir, 'log_fid.txt'), 'a+') as f:
                    f.write(f"{i:07d}; sample: {float(fid_sa):.4f}; rec_fake: {float(fid_sr):.4f}; rec_real: {float(fid_re):.4f};\n")

            if i % args.save_every == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "e": e_module.state_dict(),
                        "d": d_module.state_dict(),
                        "d2": d2_module.state_dict() if args.decouple_d else None,
                        "g_ema": g_ema.state_dict(),
                        "e_ema": e_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "d2_optim": d2_optim.state_dict() if args.decouple_d else None,
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
                        "d2": d2_module.state_dict() if args.decouple_d else None,
                        "g_ema": g_ema.state_dict(),
                        "e_ema": e_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "d2_optim": d2_optim.state_dict() if args.decouple_d else None,
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
    parser.add_argument("--arch", type=str, default='stylegan2', help="model architectures (stylegan2 | swagan)")
    parser.add_argument("--dataset", type=str, default='multires')
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--sample_cache", type=str, default=None)
    parser.add_argument("--name", type=str, help="experiment name", default='default_exp')
    parser.add_argument("--log_root", type=str, help="where to save training logs", default='logs')
    parser.add_argument("--log_every", type=int, default=100, help="save samples every # iters")
    parser.add_argument("--save_every", type=int, default=1000, help="save checkpoints every # iters")
    parser.add_argument("--save_latest_every", type=int, default=200, help="save latest checkpoints every # iters")
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
        default=8,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--which_encoder", type=str, default='style')
    parser.add_argument("--which_latent", type=str, default='w_tied')
    parser.add_argument("--stddev_group", type=int, default=1)
    parser.add_argument("--use_wscale", action='store_true', help="whether to use `wscale` layer in idinvert encoder")
    parser.add_argument("--vgg_ckpt", type=str, default="vgg16.pth")
    parser.add_argument("--output_layer_idx", type=int, default=23)
    parser.add_argument("--lambda_vgg", type=float, default=0)
    parser.add_argument("--lambda_adv", type=float, default=0)
    parser.add_argument("--lambda_pix", type=float, default=0, help="recon loss on pixel (x)")
    parser.add_argument("--lambda_fake_d", type=float, default=1.0)
    parser.add_argument("--lambda_rec_d", type=float, default=0, help="d1, recon of real image")
    parser.add_argument("--lambda_fake_g", type=float, default=1.0)
    parser.add_argument("--lambda_rec_w", type=float, default=1, help="recon sampled w")
    parser.add_argument("--pix_loss", type=str, default='l2')
    parser.add_argument("--no_joint", action='store_true', help="do not update generator with encoder")
    parser.add_argument("--inception", type=str, default=None, help="path to precomputed inception embedding")
    parser.add_argument("--eval_every", type=int, default=1000, help="interval of metric evaluation")
    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument("--n_sample_fid", type=int, default=50000, help="number of the samples for calculating FID")
    parser.add_argument("--nframe_num", type=int, default=5)
    parser.add_argument("--decouple_d", action='store_true', help="separate discriminator in ALAE")
    parser.add_argument("--n_step_d", type=int, default=1)
    parser.add_argument("--n_step_e", type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--e_ckpt", type=str, default=None, help="path to the checkpoint of encoder")
    parser.add_argument("--g_ckpt", type=str, default=None, help="path to the checkpoint of generator")
    parser.add_argument("--d_ckpt", type=str, default=None, help="path to the checkpoint of discriminator")
    parser.add_argument("--d2_ckpt", type=str, default=None, help="path to the checkpoint of discriminator2")
    parser.add_argument("--train_from_scratch", action='store_true')
    parser.add_argument("--limit_train_batches", type=float, default=1)
    parser.add_argument("--g_decay", type=float, default=None, help="g decay factor")
    parser.add_argument("--n_mlp_g", type=int, default=8)
    parser.add_argument("--n_mlp_d", type=int, default=3)
    parser.add_argument("--lr_mlp_g", type=float, default=0.01)
    parser.add_argument("--lr_mlp_d", type=float, default=0.1)
    parser.add_argument("--ema_kimg", type=int, default=10, help="Half-life of the exponential moving average (EMA) of generator weights.")
    parser.add_argument("--ema_rampup", type=float, default=None, help="EMA ramp-up coefficient.")
    parser.add_argument("--no_ema_e", action='store_true')
    parser.add_argument("--no_ema_g", action='store_true')
    parser.add_argument("--which_metric", type=str, nargs='*', default=['fid_sample'])
    parser.add_argument("--use_adaptive_weight", action='store_true', help="adaptive weight borrowed from VQGAN")
    parser.add_argument("--disc_iter_start", type=int, default=30000)
    parser.add_argument("--use_pixelnorm", action='store_true')
    parser.add_argument("--no_detach_style", action='store_true')
    parser.add_argument("--which_phi_e", type=str, default='lin2')
    parser.add_argument("--which_phi_d", type=str, default='lin2', help="which_phi of d2, if decouple_d")
    parser.add_argument("--latent_space", type=str, default='w', help="latent space (w | p | pn | z)")

    args = parser.parse_args()
    util.seed_everything()
    args.device = device

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.n_latent = int(np.log2(args.size)) * 2 - 2
    args.latent = 512
    if args.which_latent == 'w_plus':
        args.latent_full = args.latent * args.n_latent
    elif args.which_latent == 'w_tied':
        args.latent_full = args.latent
    else:
        raise NotImplementedError

    assert(
        (not args.use_adaptive_weight) or 
        (not args.no_joint and args.lambda_adv > 0 and args.lambda_pix + args.lambda_vgg > 0)
    )
    assert(args.latent_space in ['w', 'z'])

    args.start_iter = 0
    args.iter += 1
    util.set_log_dir(args)
    util.print_args(parser, args)

    from model import Generator, LatentDiscriminator

    generator = Generator(
        args.size, args.latent, args.n_mlp_g, lr_mlp=args.lr_mlp_g,
        channel_multiplier=args.channel_multiplier,
    ).to(device)
    discriminator = LatentDiscriminator(
        latent_dim=args.latent_full, n_mlp=args.n_mlp_d, lr_mlp=args.lr_mlp_d,
        hidden_dim=512, use_pixelnorm=args.use_pixelnorm,
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp_g, lr_mlp=args.lr_mlp_g,
        channel_multiplier=args.channel_multiplier,
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

    discriminator2 = d2_optim = None
    if args.decouple_d:
        from model import Discriminator
        discriminator2 = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier, which_phi=args.which_phi_d
        ).to(device)
        d2_optim = optim.Adam(
            discriminator2.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

    # Define Encoder
    if args.which_encoder == 'style':
        from model import Encoder
        encoder = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
            which_latent=args.which_latent, which_phi=args.which_phi_e,
            stddev_group=args.stddev_group).to(device)
        e_ema = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
            which_latent=args.which_latent, which_phi=args.which_phi_e,
            stddev_group=args.stddev_group).to(device)
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

        generator.load_state_dict(ckpt["g"])
        encoder.load_state_dict(ckpt["e"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        e_ema.load_state_dict(ckpt["e_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        e_optim.load_state_dict(ckpt["e_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

        if discriminator2 is not None:
            discriminator2.load_state_dict(ckpt["d2"])
            d2_optim.load_state_dict(ckpt["d2_optim"])

    elif not args.train_from_scratch:
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
            print("load d2 model:", args.d2_ckpt)
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
        if discriminator2 is not None:
            discriminator2 = nn.parallel.DistributedDataParallel(
                discriminator2,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )

    dataset = get_image_dataset(args, args.dataset, args.path, train=True)
    if args.limit_train_batches < 1:
        indices = torch.randperm(len(dataset))[:int(args.limit_train_batches * len(dataset))]
        dataset1 = data.Subset(dataset, indices)
    else:
        dataset1 = dataset
    loader = data.DataLoader(
        dataset1,
        batch_size=args.batch,
        sampler=data_sampler(dataset1, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
    )
    # A subset of length args.n_sample_fid for FID evaluation
    loader2 = None
    if args.eval_every > 0:
        indices = torch.randperm(len(dataset))[:args.n_sample_fid]
        dataset2 = data.Subset(dataset, indices)
        loader2 = data.DataLoader(dataset2, batch_size=64, num_workers=4, shuffle=False)
        if args.sample_cache is not None:
            load_real_samples(args, sample_data(loader2))

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=args.name)
    util.print_models([generator, discriminator, encoder], args)

    train(
        args, loader, loader2, generator, encoder, discriminator, discriminator2,
        vggnet, g_optim, e_optim, d_optim, d2_optim, g_ema, e_ema, device
    )
