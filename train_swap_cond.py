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
    if args.cache is not None:
        npy_path = os.path.splitext(args.cache)[0] + f"_real_{args.n_sample}.npy"
    else:
        npy_path = None
    if os.path.exists(npy_path):
        sample_x = torch.from_numpy(np.load(npy_path)).to(args.device)
    else:
        sample_x = accumulate_batches(data_iter, args.n_sample).to(args.device)
        if npy_path is not None:
            np.save(npy_path, sample_x.cpu().numpy())
    return sample_x


def cross_reconstruction(encoder, generator, frames1, frames2, frames3, cond='cond1'):
    # Conditional Discriminator 1:
    # recon pair: [frame1, recon2]
    # cross pair: [frame1, cross2]
    #  real pair: [[frame1, frame2], [frame1, frame3]]
    #  fake pair: [[frame1, recon2], [frame1, cross2]]
    # ---
    # Conditional Discriminator 2:
    # recon pair: [frame1, recon2]
    # cross pair: [frame2, cross3]
    #  real pair: [[frame1, frame2], [frame2, frame3]]
    #  fake pair: [[frame1, recon2], [frame2, cross3]]
    # ---
    # Pac Discriminator:
    #  real pair: [frame1, frame2]
    #  fake pair: [recon1, cross2]
    batch = frames1.shape[0]
    if cond == 'cond1':
        w1, _ = encoder(frames1)
        w2, _ = encoder(frames2)
        delta_w = w2 - w1
        delta_w = delta_w[torch.randperm(batch),...]
        x_recon, _ = generator([w2], input_is_latent=True, return_latents=False)
        x_real = frames2
        x_cross, _ = generator([w1 + delta_w], input_is_latent=True, return_latents=False)
        recon_pair = torch.cat((frames1, x_recon), 1)
        cross_pair = torch.cat((frames1, x_cross), 1)
        real_pair12 = torch.cat((frames1, frames2), 1)
        real_pair13 = torch.cat((frames1, frames3), 1)
        fake_pair = torch.cat((recon_pair, cross_pair), 0)
        real_pair = torch.cat((real_pair12, real_pair13), 0)
    elif cond == 'cond2':
        w1, _ = encoder(frames1)
        w2, _ = encoder(frames2)
        w3, _ = encoder(frames3)
        delta_w = w3 - w2
        delta_w = delta_w[torch.randperm(batch),...]
        x_recon, _ = generator([w2], input_is_latent=True, return_latents=False)
        x_real = frames2
        x_cross, _ = generator([w2 + delta_w], input_is_latent=True, return_latents=False)
        recon_pair = torch.cat((frames1, x_recon), 1)
        cross_pair = torch.cat((frames2, x_cross), 1)
        real_pair12 = torch.cat((frames1, frames2), 1)
        real_pair23 = torch.cat((frames2, frames3), 1)
        fake_pair = torch.cat((recon_pair, cross_pair), 0)
        real_pair = torch.cat((real_pair12, real_pair23), 0)
    elif cond == 'pac':
        w1, _ = encoder(frames1)
        w2, _ = encoder(frames2)
        delta_w = w2 - w1
        delta_w = delta_w[torch.randperm(batch),...]
        x_recon, _ = generator([w1], input_is_latent=True, return_latents=False)
        x_real = frames1
        x_cross, _ = generator([w1 + delta_w], input_is_latent=True, return_latents=False)
        fake_pair = torch.cat((x_recon, x_cross), 1)
        real_pair = torch.cat((frames1, frames2), 1)
    # return real_img, fake_img, x_real, x_recon, x_cross
    return real_pair, fake_pair, x_real, x_recon, x_cross


def train(args, loader, encoder, generator, discriminator, discriminator_w,
          vggnet, pwcnet, e_optim, g_optim, g1_optim, d_optim, dw_optim,
          e_ema, g_ema, device):
    loader = sample_data(loader)
    args.toggle_grads = True
    args.augment = False

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

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 256, device)

    # sample_x = accumulate_batches(loader, args.n_sample).to(device)
    sample_x = load_real_samples(args, loader)
    sample_x1 = sample_x[:,0,...]
    sample_x2 = sample_x[:,-1,...]
    sample_idx = torch.randperm(args.n_sample)
    sample_z = torch.randn(args.n_sample, args.latent, device=device)

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
                    delta_w = w2 - w1
                    delta_w = delta_w[sample_idx,...]
                    fake_img, _ = g_ema([w1 + delta_w], input_is_latent=True, return_latents=False)
                    sample = torch.cat((sample_x2.reshape(args.n_sample//nrow, nrow, *nchw), 
                                        fake_img.reshape(args.n_sample//nrow, nrow, *nchw)), 1)
                    utils.save_image(
                        sample.reshape(2*args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-cross.png"),
                        nrow=nrow,
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
                e_eval.train()

        if i > args.iter:
            print("Done!")
            break

        frames = next(loader)  # [N, T, C, H, W]
        batch = frames.shape[0]
        frames1 = frames[:,0,...]
        selected_indices = torch.sort(torch.multinomial(torch.ones(batch, args.nframe_num-1), 2)+1, 1)[0]
        frames2 = frames[range(batch),selected_indices[:,0],...]
        frames3 = frames[range(batch),selected_indices[:,1],...]
        frames1 = frames1.to(device)
        frames2 = frames2.to(device)
        frames3 = frames3.to(device)

        # Train Discriminator
        if args.toggle_grads:
            requires_grad(encoder, False)
            requires_grad(generator, False)
            requires_grad(discriminator, True)
            requires_grad(discriminator_w, True)

        real_img, fake_img, _, _, _ = cross_reconstruction(encoder, generator, frames1, frames2, frames3, args.cond_disc)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img_aug, _ = augment(fake_img, ada_aug_p)
        else:
            real_img_aug = real_img
            fake_img_aug = fake_img
        
        fake_pred = discriminator(fake_img_aug)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        if args.lambda_gan > 0:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img, _ = generator(noise)
            if args.augment:
                fake_img, _ = augment(fake_img, ada_aug_p)
            fake_pred = discriminator(fake_img)
            fake_loss = F.softplus(fake_pred)
            d_loss += fake_loss.mean() * args.lambda_gan

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        # loss_dict["fake_score"] = fake_pred.mean()
        fake_pred1, fake_pred2 = fake_pred.chunk(2, dim=0)
        loss_dict["fake_score"] = fake_pred1.mean()
        loss_dict["hybrid_score"] = fake_pred2.mean()

        discriminator.zero_grad()
        d_loss.backward()
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
        
        # Train Discriminator_W
        if args.learned_prior and args.lambda_gan_w > 0:
            noise = mixing_noise(args.batch, args.latent, 0, device)
            fake_w = generator.get_latent(noise[0])
            real_w, _ = encoder(frames1)
            fake_pred = discriminator_w(fake_w)
            real_pred = discriminator_w(real_w)
            d_loss_w = d_logistic_loss(real_pred, fake_pred)
            dw_optim.zero_grad()
            d_loss_w.backward()
            dw_optim.step()

        # Train Encoder and Generator
        if args.toggle_grads:
            requires_grad(encoder, True)
            requires_grad(generator, True)
            requires_grad(discriminator, False)
            requires_grad(discriminator_w, False)
        pix_loss = vgg_loss = adv_loss = rec_loss = torch.tensor(0., device=device)

        _, fake_img, x_real, x_recon, x_cross = cross_reconstruction(encoder, generator, frames1, frames2, frames3, args.cond_disc)

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
            fake_feat = vggnet(x_recon) if not args.vgg_on_cross else vggnet(x_cross)
            vgg_loss = torch.mean((fake_feat - real_feat) ** 2)

        e_loss = pix_loss * args.lambda_pix + vgg_loss * args.lambda_vgg + adv_loss * args.lambda_adv

        if args.lambda_gan > 0 and not args.no_sim_opt:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img, _ = generator(noise)
            if args.augment:
                fake_img, _ = augment(fake_img, ada_aug_p)
            fake_pred = discriminator(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred)
            e_loss += g_loss * args.lambda_gan

        loss_dict["e"] = e_loss
        loss_dict["pix"] = pix_loss
        loss_dict["vgg"] = vgg_loss
        loss_dict["adv"] = adv_loss

        e_optim.zero_grad()
        g_optim.zero_grad()
        e_loss.backward()
        e_optim.step()
        g_optim.step()

        if args.learned_prior:
            g_loss_w = 0.
            if args.lambda_gan_w > 0:
                noise = mixing_noise(args.batch, args.latent, 0, device)
                fake_w = generator.get_latent(noise[0])
                fake_pred = discriminator_w(fake_w)
                g_loss_w += g_nonsaturating_loss(fake_pred) * args.lambda_gan_w
            if args.lambda_adv_w > 0:
                noise = mixing_noise(args.batch, args.latent, args.mixing, device)
                fake_img, _ = generator(noise)
                fake_pred = discriminator(fake_img)
                g_loss_w += g_nonsaturating_loss(fake_pred) * args.lambda_adv_w
            g1_optim.zero_grad()
            g_loss_w.backward()
            g1_optim.step()
        
        if args.lambda_gan > 0 and args.no_sim_opt:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img, _ = generator(noise)
            if args.augment:
                fake_img, _ = augment(fake_img, ada_aug_p)
            fake_pred = discriminator(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred) * args.lambda_gan
            generator.zero_grad()
            g_loss.backward()
            g_optim.step()
        
        g_regularize = args.lambda_gan > 0 and args.g_reg_every > 0 and i % args.g_reg_every == 0
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
        path_loss_val = loss_reduced["path"].mean().item()
        # path_length_val = loss_reduced["path_length"].mean().item()
        avg_pix_loss.update(pix_loss_val, real_img.shape[0])
        avg_vgg_loss.update(vgg_loss_val, real_img.shape[0])

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; e: {e_loss_val:.4f}; r1_d: {r1_d_val:.4f}; "
                    f"pix: {pix_loss_val:.4f}; vgg: {vgg_loss_val:.4f}; adv: {adv_loss_val:.4f}; "
                    f"path: {path_loss_val:.4f}; augment: {ada_aug_p:.4f}"
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
                        "g": g_module.state_dict(),
                        "g_ema": g_module.state_dict(),
                        "e_ema": e_eval.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
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
                        "g": g_module.state_dict(),
                        "g_ema": g_module.state_dict(),
                        "e_ema": e_eval.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
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
    parser.add_argument("--lambda_gan", type=float, default=0., help="train a gan branch?")
    parser.add_argument("--lambda_rec", type=float, default=1.0, help="recon loss on style (w)")
    parser.add_argument("--lambda_adv_w", type=float, default=0., help="adversarial loss from image discriminator")
    parser.add_argument("--lambda_gan_w", type=float, default=0., help="adversarial loss from latent discriminator")
    parser.add_argument("--lambda_mmd_w", type=float, default=0.)
    parser.add_argument("--output_layer_idx", type=int, default=23)
    parser.add_argument("--vgg_ckpt", type=str, default="pretrained/vgg16.pth")
    parser.add_argument("--which_encoder", type=str, default='style')
    parser.add_argument("--which_latent", type=str, default='w_shared')
    parser.add_argument("--stddev_group", type=int, default=4)
    parser.add_argument("--nframe_num", type=int, default=5)
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--learned_prior", action='store_true', help="learned latent prior (w)?")
    parser.add_argument("--no_sim_opt", action='store_true')
    parser.add_argument("--cond_disc", type=str, default='cond1', choices=['cond1', 'cond2', 'pac'])
    parser.add_argument("--train_from_scratch", action='store_true')
    parser.add_argument("--vgg_on_cross", action='store_true')
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
        help="path to the checkpoints to resume training",
    )
    parser.add_argument(
        "--e_ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument(
        "--g_ckpt",
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
    args.nframe_num = max(3, args.nframe_num)

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

    in_channel = 6
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
        discriminator_w = LatentDiscriminator(args.latent, args.n_mlp).to(device)
        dw_optim = optim.Adam(
            discriminator_w.parameters(),
            lr=args.lr * 1,
            betas=(0 ** 1, 0.99 ** 1),
        )

    if args.resume and args.ckpt is not None:
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

        if args.resume:
            try:
                ckpt_name = os.path.basename(args.ckpt)
                if 'iter' in ckpt:
                    args.start_iter = ckpt["iter"]
                else:
                    args.start_iter = int(os.path.splitext(ckpt_name)[0])
            except ValueError:
                pass
            encoder.load_state_dict(ckpt["e"])
            e_optim.load_state_dict(ckpt["e_optim"])

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

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=args.name)

    train(args, loader, encoder, generator, discriminator, discriminator_w, 
          vggnet, pwcnet, e_optim, g_optim, g1_optim, d_optim, dw_optim,
          e_ema, g_ema, device)
