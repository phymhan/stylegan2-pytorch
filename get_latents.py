import numpy as np
import torch
import torch.nn.functional as F
from model import Generator, Encoder
import argparse
from tqdm import tqdm
from torch.utils import data
from dataset import get_image_dataset
import pdb
st = pdb.set_trace

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Get PCA for given ckpt")
    parser.add_argument(
        "--ckpt", type=str, default='ckpt.pt',
    )
    parser.add_argument(
        "--out_prefix", type=str, default='pn',
    )
    parser.add_argument(
        "--size", type=int, default=256,
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument("--which_latent", type=str, default='w_plus')
    parser.add_argument("--stddev_group", type=int, default=1)
    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--dataset", type=str, default='multires')

    args = parser.parse_args()

    args.latent = 512
    device = "cuda"

    encoder = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
        which_latent=args.which_latent, stddev_group=args.stddev_group).to(device)
    ckpt = torch.load(args.ckpt)
    encoder.load_state_dict(ckpt["e_ema"])
    encoder.eval()

    dataset = get_image_dataset(args, args.dataset, args.path, train=True)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data.SequentialSampler(dataset),
        drop_last=True,
    )

    latents = []
    pbar = tqdm(loader)
    with torch.no_grad():
        for real_img in pbar:
            if isinstance(real_img, (list, tuple)):
                real_img = real_img[0]
            real_img = real_img.to(device)
            latent_real, _ = encoder(real_img)
            latents.append(latent_real.cpu().numpy())
    latents = np.concatenate(latents, axis=0)

    np.savez(f"{args.out_prefix}_latents.npz", **{'latents': latents})
