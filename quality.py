"""Classement de toutes les images du bucket par qualité (0 token : calcul pur).

Mesures sur le mur et l'enrochement (zones REF_ZONES, toujours hors de l'eau) :
  luminosite : moyenne (0-255)          -> images noires
  contraste  : écart-type de luminosité -> brume
  nettete    : variance du laplacien    -> brume, buée, flou
Score : moyenne des rangs (percentiles) de contraste et netteté ; les images noires ont score 0.

Usage : python quality.py   -> télécharge ce qui manque dans data/raw/ et écrit annotations/qualite.csv
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace

from annotation_tools import DARK_MEAN, VY0, VY1
from bucket import DATA_DIR, ROOT, Bucket

OUT = ROOT / "annotations" / "qualite.csv"
CATEGORIES = ["très brumeuse", "brumeuse", "moyenne", "correcte", "nette"]  # quintiles des non-noires


# Zones toujours hors de l'eau et identiques d'une image à l'autre : la qualité mesurée ne dépend
# pas du niveau de la rivière (seulement de la brume, de la lumière, du flou).
REF_ZONES = [(520, 250, 1460, 620),   # mur en béton (x0, y0, x1, y1)
             (0, 360, 520, 600)]      # enrochement à gauche du mur


def measures(path: Path) -> dict:
    lum = np.asarray(Image.open(path).convert("L"), dtype=float)
    zones = [gaussian_filter(lum[y0:y1, x0:x1], 1.0) for x0, y0, x1, y1 in REF_ZONES]  # retire le bruit capteur
    mean = np.mean([z.mean() for z in zones])
    # normalisés par la luminosité : une image à l'ombre n'est pas pour autant brumeuse
    contrast = np.mean([z.std() for z in zones]) / (mean + 1)
    sharp = np.mean([laplace(z).var() for z in zones]) / (mean + 1) ** 2
    return {"luminosite": lum.mean(), "contraste": contrast, "nettete": sharp}


def main():
    b = Bucket()
    objs = [o for o in b.list_images() if "_permission-test" not in o["Key"]]
    print(len(objs), "images, téléchargement de ce qui manque...")
    with ThreadPoolExecutor(16) as ex:
        paths = list(ex.map(lambda o: b.download([o])[0], objs))
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(measures, paths))

    for o, r in zip(objs, rows):
        p = o["Key"].split("/")
        r.update(key=o["Key"], datetime=f"{p[1]}-{p[2]}-{p[3]}T{p[4][:2]}:{p[4][2:4]}:{p[4][4:6]}")
        r["noire"] = r["luminosite"] < DARK_MEAN

    clear = [r for r in rows if not r["noire"]]
    for k in ("contraste", "nettete"):
        v = np.array([r[k] for r in clear])
        for r, pct in zip(clear, v.argsort().argsort() / (len(v) - 1)):
            r["pct_" + k] = pct
    for r in rows:
        if r["noire"]:
            r["score"], r["categorie"] = 0.0, "noire"
        else:
            r["score"] = (r["pct_contraste"] + r["pct_nettete"]) / 2
    for i, r in enumerate(sorted(clear, key=lambda r: r["score"])):
        r["categorie"] = CATEGORIES[min(i * 5 // len(clear), 4)]

    # du pire au meilleur ; les noires triées par luminosité
    rows.sort(key=lambda r: (r["score"], r["luminosite"]))
    fields = ["rang", "datetime", "key", "categorie", "score", "luminosite", "contraste", "nettete"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"rang": i, "datetime": r["datetime"], "key": r["key"], "categorie": r["categorie"],
                        "score": round(r["score"], 4), "luminosite": round(r["luminosite"], 1),
                        "contraste": round(r["contraste"], 4), "nettete": round(r["nettete"], 6)})
    counts = {c: sum(r["categorie"] == c for r in rows) for c in ["noire"] + CATEGORIES}
    print("écrit", OUT, counts)


if __name__ == "__main__":
    main()
