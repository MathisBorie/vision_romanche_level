"""Projette quelques images sur la base POD (annotations/pod.npz) et les reconstruit avec k modes.

Usage : python projection_pod.py   -> data/preview/projection_pod.png
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyse_spectrale import PREVIEW, W0, load_band
from annotation_tools import CROP_Y0, CROP_Y1
from bucket import ROOT

KS = [1, 2, 6, 20, 50]


def main():
    pod = np.load(ROOT / "annotations" / "pod.npz")
    mean, modes, energy = pod["mean"], pod["modes"], pod["energy"]
    h, w = mean.shape
    B = modes.reshape(len(modes), -1)                     # modes orthonormés, une ligne par mode
    by_rank = {r["rang"]: r for r in csv.DictReader(open(ROOT / "annotations" / "qualite.csv"))}
    sel = [("nette 3 nov", "romanche-01/2026/11/03/104512_02821fd2_s.jpg"),
           ("crue 30 mai", "romanche-01/2026/05/30/113001_6f159d1d_s.jpg"),
           ("5 juillet", "romanche-01/2026/07/05/110011_82f0a816_s.jpg"),
           ("brumeuse 16 sept", "romanche-01/2026/09/16/110012_8e17c446_s.jpg"),
           ("brouillard " + by_rank["1094"]["datetime"][5:10], by_rank["1094"]["key"]),
           ("caméra déplacée 3 juin", by_rank["5412"]["key"])]
    ext = [0, W0, CROP_Y1, CROP_Y0]
    fig, ax = plt.subplots(len(sel), len(KS) + 3, figsize=(3.3 * (len(KS) + 3), 1.75 * len(sel)))
    for i, (t, k) in enumerate(sel):
        x = load_band(k, 8, gray=True).reshape(-1).astype(np.float64)
        d = x - mean.reshape(-1)
        a = B @ d                                         # coefficients de projection a_k = <x - moyenne, phi_k>
        ax[i, 0].imshow(x.reshape(h, w), cmap="gray", vmin=0, vmax=1, extent=ext)
        ax[i, 0].set_title(f"{t} : original", fontsize=8, loc="left")
        for j, K in enumerate(KS, 1):
            rec = mean.reshape(-1) + a[:K] @ B[:K]
            err = np.linalg.norm(d - a[:K] @ B[:K]) / np.linalg.norm(d)
            ax[i, j].imshow(rec.reshape(h, w), cmap="gray", vmin=0, vmax=1, extent=ext)
            ax[i, j].set_title(f"{K} mode{'s' * (K > 1)} : erreur {100 * err:.0f} %", fontsize=8, loc="left")
        res = d - a @ B
        v = np.abs(res).max()
        ax[i, -2].imshow(res.reshape(h, w), cmap="RdBu_r", vmin=-v, vmax=v, extent=ext)
        ax[i, -2].set_title(f"résidu après {len(B)} modes", fontsize=8, loc="left")
        ax[i, -1].bar(range(1, 11), a[:10], color=["C3" if c < 0 else "C0" for c in a[:10]])
        ax[i, -1].set_title("a1..a10", fontsize=8, loc="left"); ax[i, -1].axhline(0, c="k", lw=.5)
        ax[i, -1].tick_params(labelsize=6)
        print(f"{t:28s}", " ".join(f"{c:7.1f}" for c in a[:6]))
    for a_ in ax[:, :-1].ravel():
        a_.set_xticks([]); a_.tick_params(labelsize=6)
    plt.tight_layout(); plt.savefig(PREVIEW / "projection_pod.png", dpi=65)


if __name__ == "__main__":
    main()
