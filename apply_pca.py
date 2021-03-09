import argparse

import torch
from torchvision import utils
import numpy as np
from model import Generator


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="pca",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "--factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )
    parser.add_argument("--w_plus", action='store_true')
    parser.add_argument("--latent_index", type=int, default=[0], nargs='+')

    args = parser.parse_args()

    eigvec = torch.from_numpy(np.load(args.factor)["comp"]).to(device)
    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)

    args.n_latent = int(np.log2(args.size)) * 2 - 2
    if args.w_plus:
        latent = latent.unsqueeze(1).repeat(1, args.n_latent, 1)
        direction = torch.zeros(args.n_sample, args.n_latent, 512, device=args.device)
        print(f"latent index: {args.latent_index}")
        out_suffix = f"_wp-{'-'.join(map(str, args.latent_index))}"
        for j in args.latent_index:
            direction[:, j, :] += args.degree * eigvec[:, args.index].unsqueeze(0)
    else:
        out_suffix = ""
        direction = args.degree * eigvec[:, args.index].unsqueeze(0)

    img, _ = g(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img1, _ = g(
        [latent + direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img2, _ = g(
        [latent - direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )

    grid = utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"{args.out_prefix}_index-{args.index}_degree-{args.degree}{out_suffix}.png",
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    )
