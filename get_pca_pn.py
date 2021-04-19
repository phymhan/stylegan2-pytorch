import numpy as np
import torch
import torch.nn.functional as F
from model import Generator
import argparse
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
    args = parser.parse_args()

    device = "cuda"

    g = Generator(args.size, 512, 8, 2).to(device)
    g.load_state_dict(torch.load(args.ckpt)["g_ema"])
    g.eval()

    Z = []
    W = []
    P = []
    with torch.no_grad():
        for i in range(100):
            z = torch.randn(1000, 512).to(device)
            w = g.style(z)
            Z.append(z.cpu().numpy())
            W.append(w.cpu().numpy())
            P.append(F.leaky_relu(w, 5.0).cpu().numpy())
    P = np.concatenate(P, axis=0)
    W = np.concatenate(W, axis=0)
    Z = np.concatenate(Z, axis=0)

    # P is of shape [N, 512]
    P_mu = np.mean(P, axis=0)
    P_cov = np.cov(P - P_mu, rowvar=False)
    S, C = np.linalg.eig(P_cov)

    # Save PCA states
    pca_state = {'Lambda': np.sqrt(S), 'C': C, 'mu': P_mu}
    np.savez(f"{args.out_prefix}_PN_PCA.npz", **pca_state)

    # Save features: z, w, p, and pn
    Pn = np.matmul(P - P_mu, C) / np.sqrt(S)
    np.save(f'{args.out_prefix}_p.npy', P)
    np.save(f'{args.out_prefix}_pn.npy', Pn)
    np.save(f'{args.out_prefix}_w.npy', W)
    np.save(f'{args.out_prefix}_z.npy', Z)
