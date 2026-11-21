"""Filtres de Fourier, écart à l'image moyenne et POD sur la bande utile (CROP_Y0-CROP_Y1).

Usage : python analyse_spectrale.py   -> figures dans data/preview/
  filtres.png : passe-bas, passe-bande et image - moyenne sur quelques images
  pod.png     : image moyenne, énergie des modes, premiers modes et leurs coefficients dans le temps
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from annotation_tools import CROP_Y0, CROP_Y1
from bucket import DATA_DIR, ROOT

PREVIEW = ROOT / "data" / "preview"
W0 = 1919


def load_band(key: str, scale: int, gray=False) -> np.ndarray:
    """Bande CROP_Y0-CROP_Y1, décodée directement à 1/scale (décodage JPEG réduit, rapide)."""
    im = Image.open(DATA_DIR / key)
    im.draft("L" if gray else "RGB", (im.size[0] // scale, im.size[1] // scale))
    s = W0 / im.size[0]  # échelle réellement obtenue
    im = im.convert("L" if gray else "RGB").crop((0, round(CROP_Y0 / s), im.size[0], round(CROP_Y1 / s)))
    return np.asarray(im, dtype=np.float32) / 255


def fourier_filter(im: np.ndarray, H) -> np.ndarray:
    """Applique un filtre H(r) (r en cycles/px de l'image traitée), avec prolongement miroir."""
    h, w = im.shape[:2]
    pad = ((h, h), (w, w)) + ((0, 0),) * (im.ndim - 2)
    p = np.pad(im, pad, mode="symmetric")
    r = np.hypot(np.fft.fftfreq(p.shape[0])[:, None], np.fft.fftfreq(p.shape[1])[None, :])
    Hr = H(r)[(...,) + (None,) * (im.ndim - 2)]
    out = np.real(np.fft.ifft2(np.fft.fft2(p, axes=(0, 1)) * Hr, axes=(0, 1)))
    return out[h:2 * h, w:2 * w]


def lowpass(fc):            # gaussien
    return lambda r: np.exp(-(r / fc) ** 2 / 2)


def bandpass(f1, f2):       # différence de deux gaussiennes : garde ~[f1, f2]
    return lambda r: np.exp(-(r / f2) ** 2 / 2) - np.exp(-(r / f1) ** 2 / 2)


def show_signed(ax, x, title, q=99):
    """Affiche un champ signé centré sur 0 (gris = 0)."""
    v = np.percentile(np.abs(x), q) + 1e-6
    ax.imshow(np.clip(x / (2 * v) + 0.5, 0, 1), extent=[0, W0, CROP_Y1, CROP_Y0],
              cmap="gray" if x.ndim == 2 else None)
    ax.set_title(title, fontsize=9, loc="left")


def main():
    rows = list(csv.DictReader(open(ROOT / "annotations" / "qualite.csv")))
    day = [r for r in rows if r["categorie"] != "noire" and "_permission" not in r["key"]]
    print(len(day), "images de jour")

    # --- image moyenne (couleur, 1/2) ---
    S = 2
    with ThreadPoolExecutor(8) as ex:
        acc = None
        for b in ex.map(lambda r: load_band(r["key"], S), day):
            acc = b.astype(np.float64) if acc is None else acc + b
    mean = (acc / len(day)).astype(np.float32)

    # --- figure filtres ---
    by_rank = {r["rang"]: r for r in rows}
    sel = [("nette 3 nov", "romanche-01/2026/11/03/104512_02821fd2_s.jpg"),
           ("crue 30 mai", "romanche-01/2026/05/30/113001_6f159d1d_s.jpg"),
           ("5 juillet", "romanche-01/2026/07/05/110011_82f0a816_s.jpg"),
           ("brumeuse 16 sept", "romanche-01/2026/09/16/110012_8e17c446_s.jpg"),
           ("brumeuse 16 oct", "romanche-01/2026/10/16/110006_0825eea1_s.jpg"),
           ("brouillard " + by_rank["1094"]["datetime"][5:10], by_rank["1094"]["key"])]
    # fréquences en cycles/px de l'image d'origine ; x S car l'image traitée est réduite
    cols = [("original", None),
            ("passe-bas fc=0.03 (>33 px)", ("lp", 0.03)),
            ("passe-bas fc=0.01 (>100 px)", ("lp", 0.01)),
            ("passe-bande 0.01-0.03 (33-100 px)", ("bp", 0.01, 0.03)),
            ("passe-bande 0.03-0.1 (10-33 px)", ("bp", 0.03, 0.1)),
            ("image - image moyenne", ("mean",))]
    fig, ax = plt.subplots(len(sel), len(cols), figsize=(4.2 * len(cols), 2.0 * len(sel)))
    for i, (t, k) in enumerate(sel):
        im = load_band(k, S)
        for j, (ct, op) in enumerate(cols):
            a = ax[i, j]
            if op is None:
                a.imshow(im, extent=[0, W0, CROP_Y1, CROP_Y0]); a.set_title(f"{t} : {ct}", fontsize=9, loc="left")
            elif op[0] == "lp":
                a.imshow(np.clip(fourier_filter(im, lowpass(op[1] * S)), 0, 1), extent=[0, W0, CROP_Y1, CROP_Y0])
                a.set_title(ct, fontsize=9, loc="left")
            elif op[0] == "bp":
                show_signed(a, fourier_filter(im, bandpass(op[1] * S, op[2] * S)).mean(axis=2), ct)
            else:
                show_signed(a, im - mean, ct)
            a.set_xticks([]); a.tick_params(labelsize=6)
    plt.tight_layout(); plt.savefig(PREVIEW / "filtres.png", dpi=60); plt.close()

    # --- POD (niveaux de gris, 1/8) ---
    S8 = 8
    with ThreadPoolExecutor(8) as ex:
        X = np.stack(list(ex.map(lambda r: load_band(r["key"], S8, gray=True), day)))
    n, h, w = X.shape
    Xm = X.reshape(n, -1).astype(np.float64)
    xbar = Xm.mean(axis=0)
    A = Xm - xbar
    # méthode des snapshots : valeurs propres de A A^T (n x n), plus petite que A^T A
    lam, V = np.linalg.eigh(A @ A.T)
    lam, V = lam[::-1], V[:, ::-1]
    sv = np.sqrt(np.clip(lam, 0, None))
    nm = 50                                       # modes gardés (seuls les 6 premiers sont tracés)
    modes = (A.T @ V[:, :nm]) / sv[:nm]           # modes spatiaux normés
    coefs = V[:, :nm] * sv[:nm]                   # coefficients temporels a_k(t)
    energy = lam / lam.sum()
    np.savez_compressed(ROOT / "annotations" / "pod.npz", keys=np.array([r["key"] for r in day]),
                        mean=xbar.reshape(h, w), modes=modes.T.reshape(nm, h, w), coefs=coefs, energy=energy)

    dates = [datetime.fromisoformat(r["datetime"]) for r in day]
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(5, 4)
    a = fig.add_subplot(gs[0, :2]); a.imshow(xbar.reshape(h, w), cmap="gray", extent=[0, W0, CROP_Y1, CROP_Y0])
    a.set_title(f"image moyenne ({n} images de jour)", loc="left")
    a = fig.add_subplot(gs[0, 2:])
    a.plot(np.arange(1, 31), 100 * energy[:30], "o-", label="énergie du mode (%)")
    a.plot(np.arange(1, 31), 100 * np.cumsum(energy[:30]), "s-", label="cumulée (%)")
    a.set_xlabel("mode"); a.grid(alpha=.3); a.legend(); a.set_title("spectre POD", loc="left")
    for k in range(6):
        a = fig.add_subplot(gs[1 + k // 2, (k % 2) * 2])
        m = modes[:, k].reshape(h, w)
        v = np.abs(m).max()
        a.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v, extent=[0, W0, CROP_Y1, CROP_Y0])
        a.set_title(f"mode {k + 1} ({100 * energy[k]:.1f} %)", loc="left", fontsize=10)
        a = fig.add_subplot(gs[1 + k // 2, (k % 2) * 2 + 1])
        a.scatter(dates, coefs[:, k], s=2, c=[d.hour for d in dates], cmap="viridis")
        a.set_title(f"a{k + 1}(t)  (couleur = heure)", loc="left", fontsize=10); a.grid(alpha=.3)
        a.tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig(PREVIEW / "pod.png", dpi=60); plt.close()
    print("énergie 6 premiers modes (%) :", np.round(100 * energy[:6], 1))


if __name__ == "__main__":
    main()
