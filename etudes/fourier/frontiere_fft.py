"""Détection automatique de la frontière eau/berge par FFT ligne par ligne (0 token).

Principe : la berge (rochers, galets) a beaucoup plus d'énergie aux hautes fréquences horizontales que l'eau.
Pour chaque ligne y, on calcule l'énergie de la FFT (le long de x) dans la bande [F_LO, F_HI] cycles/px ;
la frontière est la ligne où cette énergie chute le plus (marche descendante en allant vers le bas).
On le fait sur toute la largeur (y_global) et sur 10 tronçons centrés sur les colonnes C1..C10.

Usage :
  python frontiere_fft.py ref     -> compare aux relevés manuels (annotations/manuel.csv)
  python frontiere_fft.py         -> toutes les images de bonne qualité -> annotations/frontiere_fft.csv
"""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from annotation_tools import CROP_Y0, CROP_Y1, XS
from bucket import DATA_DIR, ROOT

F_LO, F_HI = 0.1, 0.3      # bande de fréquences (cycles/px) : texture des galets, peu présente sur l'eau
SEG = 190                  # largeur d'un tronçon de colonnes (px)
W = 40                     # demi-fenêtre (px) pour mesurer la marche
SEARCH = (600, 1150)       # plage de y où l'on cherche la frontière (dans la bande CROP)
A1_MAX_NEG = -25           # coefficient POD a1 en dessous duquel le cadrage est suspect


def band_energy(g: np.ndarray) -> np.ndarray:
    """log de l'énergie par ligne dans [F_LO, F_HI] (FFT 1D le long de x, fenêtre de Hann)."""
    g = g - g.mean(axis=1, keepdims=True)
    F = np.abs(np.fft.rfft(g * np.hanning(g.shape[1]), axis=1)) ** 2
    f = np.fft.rfftfreq(g.shape[1])
    return np.log(F[:, (f >= F_LO) & (f <= F_HI)].mean(axis=1) + 1e-9)


def find_step(e: np.ndarray) -> tuple[int, float]:
    """y de la plus forte chute d'énergie, et son amplitude (contraste berge/eau, en log)."""
    e = gaussian_filter1d(e, 3)
    best, amp = None, -np.inf
    for y in range(SEARCH[0] - CROP_Y0, SEARCH[1] - CROP_Y0):
        d = e[max(0, y - W):y].mean() - e[y:y + W].mean()
        if d > amp:
            best, amp = y, d
    return best + CROP_Y0, float(amp)


def detect(key: str) -> dict:
    g = np.asarray(Image.open(DATA_DIR / key).convert("L").crop((0, CROP_Y0, 1919, CROP_Y1)), dtype=float) / 255
    y, amp = find_step(band_energy(g))
    r = {"y_global": y, "contraste_global": round(amp, 3)}
    for x in XS:
        seg = g[:, max(0, x - SEG // 2):x + SEG // 2]
        yx, ax = find_step(band_energy(seg))
        r[f"y_x{x}"], r[f"c_x{x}"] = yx, round(ax, 3)
    ys = [r[f"y_x{x}"] for x in XS]
    r["y_moyen"] = round(float(np.mean(ys)), 1)
    r["y_median"] = float(np.median(ys))
    r["dispersion"] = round(float(np.std(ys)), 1)  # forte dispersion = frontière incohérente entre tronçons
    return r


def run_ref():
    ref = list(csv.DictReader(open(ROOT / "donnees" / "annotations_anciennes" / "manuel.csv")))
    for m in ref:
        r = detect(m["key"])
        man = np.array([float(m[f"x{x}"]) for x in XS])
        auto = np.array([r[f"y_x{x}"] for x in XS])
        print(f"{m['datetime'][:10]}  manuel {man.mean():6.1f}  auto moyen {r['y_moyen']:6.1f}  "
              f"global {r['y_global']}  écart médian/colonne {np.median(np.abs(auto - man)):5.1f}  "
              f"contraste {r['contraste_global']:.2f}")
        print("   manuel", man.astype(int).tolist())
        print("   auto  ", auto.astype(int).tolist())


def run_all():
    pod = np.load(ROOT / "data" / "pod.npz")
    a1 = dict(zip(pod["keys"], pod["coefs"][:, 0]))
    rows = [r for r in csv.DictReader(open(ROOT / "donnees" / "annotations_anciennes" / "qualite.csv"))
            if r["categorie"] in ("nette", "correcte")]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(detect, [r["key"] for r in rows], chunksize=16))
    out = []
    for q, r in zip(rows, res):
        cadrage_ok = a1.get(q["key"], 0) >= A1_MAX_NEG
        out.append({"datetime": q["datetime"], "key": q["key"], "categorie": q["categorie"],
                    "cadrage": "ok" if cadrage_ok else "suspect", **r})
    out.sort(key=lambda r: r["datetime"])
    fields = list(out[0])
    with open(ROOT / "data" / "frontiere_fft.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(len(out), "images,", sum(r["cadrage"] == "suspect" for r in out), "cadrage suspect")


if __name__ == "__main__":
    run_ref() if sys.argv[1:] == ["ref"] else run_all()
