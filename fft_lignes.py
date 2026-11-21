"""Transformée de Fourier 1D ligne par ligne sur la bande utile, affichée comme une image :
abscisse = fréquence spatiale horizontale fx, ordonnée = ligne y, couleur = amplitude (log).

Usage : python fft_lignes.py   -> data/preview/fft_lignes.png
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from annotation_tools import CROP_Y0, CROP_Y1
from bucket import DATA_DIR, ROOT


def row_spectrum(g: np.ndarray):
    """|FFT| de chaque ligne (moyenne de la ligne retirée, fenêtre de Hann contre les effets de bord)."""
    g = g - g.mean(axis=1, keepdims=True)
    win = np.hanning(g.shape[1])
    F = np.abs(np.fft.rfft(g * win, axis=1))
    return np.fft.rfftfreq(g.shape[1]), np.log10(F + 1e-3)


def main():
    by_rank = {r["rang"]: r for r in csv.DictReader(open(ROOT / "annotations" / "qualite.csv"))}
    sel = [("nette 3 nov", "romanche-01/2026/11/03/104512_02821fd2_s.jpg"),
           ("crue 30 mai", "romanche-01/2026/05/30/113001_6f159d1d_s.jpg"),
           ("5 juillet", "romanche-01/2026/07/05/110011_82f0a816_s.jpg"),
           ("brumeuse 16 sept", "romanche-01/2026/09/16/110012_8e17c446_s.jpg"),
           ("brumeuse 16 oct", "romanche-01/2026/10/16/110006_0825eea1_s.jpg"),
           ("brouillard " + by_rank["1094"]["datetime"][5:10], by_rank["1094"]["key"])]
    spectra = []
    for t, k in sel:
        im = Image.open(DATA_DIR / k).convert("RGB").crop((0, CROP_Y0, 1919, CROP_Y1))
        g = np.asarray(im.convert("L"), dtype=float) / 255
        f, S = row_spectrum(g)
        spectra.append((t, np.asarray(im), f, S))
    vmin, vmax = np.percentile(np.concatenate([s[3].ravel() for s in spectra]), [2, 99.8])  # même échelle partout

    fig, ax = plt.subplots(len(sel), 2, figsize=(18, 3.3 * len(sel)), gridspec_kw={"width_ratios": [1.4, 1]})
    for i, (t, im, f, S) in enumerate(spectra):
        ax[i, 0].imshow(im, extent=[0, 1919, CROP_Y1, CROP_Y0], aspect="auto")
        ax[i, 0].set_title(f"{t} : bande y {CROP_Y0}-{CROP_Y1}", loc="left", fontsize=10)
        m = ax[i, 1].imshow(S, extent=[f[0], f[-1], CROP_Y1, CROP_Y0], aspect="auto", cmap="inferno",
                            vmin=vmin, vmax=vmax)
        ax[i, 1].set_title("FFT de chaque ligne : |F(fx)| (log10)", loc="left", fontsize=10)
        ax[i, 1].set_xlabel("fx (cycles/px)   [0.01 = 100 px, 0.1 = 10 px]", fontsize=8)
        ax[i, 0].set_ylabel("y (px)")
        plt.colorbar(m, ax=ax[i, 1], fraction=0.03)
    plt.tight_layout(); plt.savefig(ROOT / "data" / "preview" / "fft_lignes.png", dpi=60)


if __name__ == "__main__":
    main()
