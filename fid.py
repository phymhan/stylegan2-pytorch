import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from model import Generator
from calc_inception import load_patched_inception_v3
import pdb
st = pdb.set_trace

@torch.no_grad()
def extract_feature_from_samples(
    generator, inception, truncation, truncation_latent, batch_size, n_sample, device,
    prior=None, samples=None, verbose=False, n_classes=-1,
):
    # generator is conditional if n_classes > 0
    conditional = n_classes > 0
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid] if resid > 0 else [batch_size] * n_batch
    features = []
    cnt = 0
    pbar = tqdm(batch_sizes) if verbose else batch_sizes
    for batch in pbar:
        if samples is not None:
            img = samples[cnt:cnt+batch,...]
            cnt += batch
        elif conditional:
            latent = torch.randn(batch, 512, device=device)
            fake_labels = torch.empty(batch, dtype=torch.long).random_(n_classes).to(device)
            img, _ = generator([latent], fake_labels,
                truncation=truncation, truncation_latent=truncation_latent)
        else:
            latent = torch.randn(batch, 512, device=device)
            if prior is None:
                img, _ = generator([latent], truncation=truncation, truncation_latent=truncation_latent)
            else:
                latent = prior(latent)
                img, _ = generator([latent], input_is_latent=True, truncation=truncation, truncation_latent=truncation_latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to("cpu"))

    features = torch.cat(features, 0)

    return features


@torch.no_grad()
def extract_feature_from_reconstruction(
    encoder, generator, inception, truncation, truncation_latent, loader, device,
    encode_z=False, mode='hybrid', shuffle_idx=None, verbose=False,
):
    # batch_size = loader.batch_size
    features = []
    pbar = tqdm(loader) if verbose else loader
    for imgs in pbar:
        if isinstance(imgs, (list, tuple)):
            imgs = imgs[0]
        imgs = imgs.to(device)
        if mode == 'recon':
            if imgs.ndim > 4:  # [N, T, C, H, W]
                imgs = imgs[:,0,...]
            w, _ = encoder(imgs)
            img, _ = generator([w], input_is_latent=not encode_z)
        elif mode == 'hybrid':
            frames1 = imgs[:,0,...]
            frames2 = imgs[:,-1,...]
            w1, _ = encoder(frames1)
            w2, _ = encoder(frames2)
            dw = w2 - w1
            if shuffle_idx is None:
                # Swap upper and lower half
                dw_ = torch.cat(dw.chunk(2, 0)[::-1], 0)
            else:  # Shuffle by shuffle_idx
                j = shuffle_idx[shuffle_idx<frames1.shape[0]] if len(shuffle_idx) > frames1.shape[0] else shuffle_idx
                dw_ = dw[j]
            img, _ = generator([w1 + dw_], input_is_latent=not encode_z)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to("cpu"))

    features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Calculate FID scores")

    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of samples to calculate mean for truncation",
    )
    parser.add_argument(
        "--batch", type=int, default=64, help="batch size for the generator"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=50000,
        help="number of the samples for calculating FID",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for generator"
    )
    parser.add_argument(
        "--inception",
        type=str,
        default=None,
        required=True,
        help="path to precomputed inception embedding",
    )
    parser.add_argument(
        "ckpt", metavar="CHECKPOINT", help="path to generator checkpoint"
    )

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)

    g = Generator(args.size, 512, 8).to(device)
    g.load_state_dict(ckpt["g_ema"])
    g = nn.DataParallel(g)
    g.eval()

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g.mean_latent(args.truncation_mean)

    else:
        mean_latent = None

    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()

    features = extract_feature_from_samples(
        g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
    ).numpy()
    print(f"extracted {features.shape[0]} features")

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    with open(args.inception, "rb") as f:
        embeds = pickle.load(f)
        real_mean = embeds["mean"]
        real_cov = embeds["cov"]

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

    print("fid:", fid)
