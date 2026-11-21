"""POD en couleur sur la bande utile (R, G, B concaténés), toutes les images de jour.

Usage : python pod_couleur.py   -> data/preview/pod_couleur.png et annotations/pod_couleur.npz
Affichage d'un mode : gris = 0 ; une teinte indique dans quel sens ce mode décale la couleur de chaque pixel.
"""
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyse_spectrale import PREVIEW, W0, load_band
from annotation_tools import CROP_Y0, CROP_Y1
from bucket import ROOT

NM = 50  # modes gardés (6 tracés)


def main():
    day = [r for r in csv.DictReader(open(ROOT / "annotations" / "qualite.csv"))
           if r["categorie"] != "noire" and "_permission" not in r["key"]]
    with ThreadPoolExecutor(8) as ex:
        X = np.stack(list(ex.map(lambda r: load_band(r["key"], 8), day)))  # (n, h, w, 3)
    n, h, w, c = X.shape
    A = X.reshape(n, -1)
    xbar = A.mean(axis=0)
    A = A - xbar
    lam, V = np.linalg.eigh((A @ A.T).astype(np.float64))     # méthode des snapshots
    lam, V = lam[::-1], V[:, ::-1]
    sv = np.sqrt(np.clip(lam[:NM], 0, None))
    modes = (A.T @ V[:, :NM].astype(np.float32)) / sv.astype(np.float32)   # (h*w*3, NM), orthonormés
    coefs = V[:, :NM] * sv
    energy = lam / lam.sum()
    np.savez_compressed(ROOT / "annotations" / "pod_couleur.npz", keys=np.array([r["key"] for r in day]),
                        mean=xbar.reshape(h, w, c), modes=modes.T.reshape(NM, h, w, c), coefs=coefs, energy=energy)

    ext = [0, W0, CROP_Y1, CROP_Y0]
    dates = [datetime.fromisoformat(r["datetime"]) for r in day]
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 4)
    a = fig.add_subplot(gs[0, :2]); a.imshow(np.clip(xbar.reshape(h, w, c), 0, 1), extent=ext, aspect="auto")
    a.set_title(f"image moyenne ({n} images de jour)", loc="left")
    a = fig.add_subplot(gs[0, 2:])
    a.plot(np.arange(1, 31), 100 * energy[:30], "o-", label="énergie du mode (%)")
    a.plot(np.arange(1, 31), 100 * np.cumsum(energy[:30]), "s-", label="cumulée (%)")
    a.set_xlabel("mode"); a.grid(alpha=.3); a.legend(); a.set_title("spectre POD couleur", loc="left")
    for k in range(6):
        m = modes[:, k].reshape(h, w, c)
        a = fig.add_subplot(gs[1 + k // 2, (k % 2) * 2])
        a.imshow(np.clip(0.5 + m / (2 * np.abs(m).max()), 0, 1), extent=ext, aspect="auto")
        a.set_title(f"mode {k + 1} ({100 * energy[k]:.1f} %)  gris = 0", loc="left", fontsize=10)
        a = fig.add_subplot(gs[1 + k // 2, (k % 2) * 2 + 1])
        a.scatter(dates, coefs[:, k], s=2, c=[d.hour for d in dates], cmap="viridis")
        a.set_title(f"a{k + 1}(t)  (couleur = heure)", loc="left", fontsize=10)
        a.grid(alpha=.3); a.tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig(PREVIEW / "pod_couleur.png", dpi=60)
    print("énergie 6 premiers modes (%) :", np.round(100 * energy[:6], 1))


if __name__ == "__main__":
    main()
