"""Compare des photos test à des photos de référence (niveau connu) par produit scalaire.

Chaque photo = vecteur (bande y 500-1200 en couleur, réduite 1/8 : 87 x 240 x 3 valeurs, entre 0 et 1).
Trois mesures entre une photo test x et une référence r :
  - produit scalaire brut   <x, r>
  - cosinus                 <x, r> / (|x| |r|)              (insensible à un facteur de luminosité global)
  - corrélation             cosinus après avoir retiré à chaque vecteur sa propre moyenne
                            et divisé par son écart-type (insensible à la luminosité ET au contraste)
Le niveau « prédit » est celui de la référence qui donne la plus grande valeur.

Usage : python produit_scalaire.py
"""
import csv

import numpy as np

from analyse_spectrale import load_band
from bucket import ROOT

refs = list(csv.DictReader(open(ROOT / "donnees" / "annotations_anciennes" / "references_niveau.csv")))
tests = list(csv.DictReader(open(ROOT / "donnees" / "annotations_anciennes" / "tests_niveau.csv")))
R = np.stack([load_band(r["key"], 8).ravel() for r in refs]).astype(np.float64)
T = np.stack([load_band(t["key"], 8).ravel() for t in tests]).astype(np.float64)
lv_ref = np.array([float(r["y_moyen"]) for r in refs])


def zscore(M):
    return (M - M.mean(axis=1, keepdims=True)) / M.std(axis=1, keepdims=True)


measures = {
    "produit scalaire brut": T @ R.T,
    "cosinus": (T / np.linalg.norm(T, axis=1, keepdims=True)) @ (R / np.linalg.norm(R, axis=1, keepdims=True)).T,
    "corrélation": zscore(T) @ zscore(R).T / T.shape[1],
}

head = "réf.  " + " ".join(f"{r['datetime'][5:10]:>8}" for r in refs)
levels = "niveau" + " ".join(f"{v:8.0f}" for v in lv_ref)
for name, M in measures.items():
    print(f"\n===== {name} =====")
    print(head); print(levels)
    for t, row in zip(tests, M):
        best = int(np.argmax(row))
        fmt = "{:8.0f}" if name == "produit scalaire brut" else "{:8.3f}"
        vals = " ".join(fmt.format(v) for v in row)
        print(f"{t['datetime'][5:10]} {vals}   -> max : réf. {refs[best]['datetime'][5:10]} "
              f"(niveau {lv_ref[best]:.0f}) | vrai niveau {float(t['y_moyen']):.0f} "
              f"| écart {lv_ref[best] - float(t['y_moyen']):+.0f} px")

# ---- figure : pour chaque photo test, la valeur de chaque mesure en fonction du niveau de la référence
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from annotation_tools import CROP_Y0, CROP_Y1

o = np.argsort(lv_ref)
fig, ax = plt.subplots(len(tests), 4, figsize=(22, 2.9 * len(tests)), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
for i, t in enumerate(tests):
    ax[i, 0].imshow(load_band(t["key"], 4), extent=[0, 1919, CROP_Y1, CROP_Y0], aspect="auto")
    ax[i, 0].axhline(float(t["y_moyen"]), color="yellow", lw=1.5)
    ax[i, 0].set_title(f"test {t['datetime'][:16].replace('T', ' ')} : niveau relevé {float(t['y_moyen']):.0f}",
                       loc="left", fontsize=10)
    for j, (name, M) in enumerate(measures.items(), 1):
        a = ax[i, j]
        a.plot(lv_ref[o], M[i, o], "o-")
        best = int(np.argmax(M[i]))
        a.plot(lv_ref[best], M[i, best], "r*", ms=14)
        a.axvline(float(t["y_moyen"]), color="k", ls="--", lw=1)
        a.set_title(f"{name} : max -> niveau {lv_ref[best]:.0f}", loc="left", fontsize=10)
        a.grid(alpha=.3)
        if i == len(tests) - 1:
            a.set_xlabel("niveau de la référence (y en px)")
plt.tight_layout()
plt.savefig(ROOT / "data" / "preview" / "produit_scalaire.png", dpi=60)
