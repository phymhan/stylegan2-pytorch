import numpy as np
from ipca import IPCAEstimator
import torch
from model import Generator
import argparse


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Get PCA for given ckpt")
    parser.add_argument(
        "--ckpt", type=str, default='ckpt.pt',
    )
    parser.add_argument(
        "--out_prefix", type=str, default='pca',
    )
    parser.add_argument(
        "--size", type=int, default=256,
    )
    args = parser.parse_args()

    device = "cuda"

    g = Generator(args.size, 512, 8, 2).to(device)
    g.load_state_dict(torch.load(args.ckpt)["g_ema"])
    g.eval()

    w = []
    with torch.no_grad():
        for i in range(100):
            z = torch.randn(1000, 512).to(device)
            w.append(g.style(z).cpu().numpy())
    w = np.concatenate(w, axis=0)

    pca = IPCAEstimator(n_components=512)
    pca.fit(w)
    comp, std, var_ratio = pca.get_components()
    pca_state = {'comp': comp, 'std': std, 'var_ratio': var_ratio}
    np.savez(f"{args.out_prefix}_pca.npz", **pca_state)
