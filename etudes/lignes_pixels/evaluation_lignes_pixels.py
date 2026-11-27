"""Précision des trois détecteurs « lignes de pixels » sur les photos test annotées.

    .venv/bin/python donnees/split_train_test.py     # d'abord : découpage train / test
    .venv/bin/python evaluation_lignes_pixels.py            # puis : évaluation

Méthodes (aucun entraînement, règles fixes) :
    - Stats RGB          Lignes_pixels.py                  (variation des moyennes / écarts-types par ligne)
    - Canny + gradient   Lignes_pixels_canny_gradient.py   (chute de la densité de contours)
    - Texture            Lignes_pixels_texture.py          (gradient du profil de variance locale, y_final)

Vérité terrain : donnees/annotations/niveau_*.csv (dernière annotation gagne, comme
encodeur/donnees.py), restreinte à split == "test" de donnees/split_train_test.csv.

Sorties :
    evaluation/predictions_lignes_pixels.csv
    figures/precision_lignes_pixels.png
"""
from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Lignes_pixels as rgb
import Lignes_pixels_canny_gradient as canny
import Lignes_pixels_texture as texture

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir())
ICI = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "images"
TRAITEMENT = ROOT / "donnees"
SORTIE_CSV = ICI / "resultats" / "predictions_lignes_pixels.csv"
SORTIE_PNG = ICI / "figures" / "precision_lignes_pixels.png"
HAUTEUR = 1753

# Rappel (tableau encodeurs vs POD, mêmes 883 photos test)
DINOV2 = {"erreur moyenne": 6.0, "erreur médiane": 5.3, "erreur max": 23.3,
          "à 5 px près": 48.5, "à 10 px près": 80.3, "à 20 px près": 99.9}


# ------------------------------------------------------------------ prédicteurs (y en px)

def pred_rgb(chemin: Path) -> int:
    return int(rgb.detect_waterline(chemin)[3])


def pred_canny(chemin: Path) -> int:
    return int(canny.detect_waterline(chemin)[5])


def pred_texture(chemin: Path) -> int:
    """Même pipeline que Lignes_pixels_texture.py : y_final (= y_grad, confirmé ou non par le step)."""
    image = cv2.imread(str(chemin))
    h = image.shape[0]
    y_min, y_max = int(h * texture.Y_MIN_RATIO), int(h * texture.Y_MAX_RATIO)
    profil, _ = texture.compute_texture_profile(image)
    y_grad, _ = texture.method_gradient(profil, y_min, y_max)
    return int(y_grad)


METHODES = {
    "rgb": ("Stats RGB\n(Lignes_pixels)", pred_rgb, rgb),
    "canny": ("Canny + gradient\n(canny_gradient)", pred_canny, canny),
    "texture": ("Texture\n(variance locale)", pred_texture, texture),
}


def _un(args):
    cle, fichier = args
    return METHODES[cle][1](IMAGES / fichier)


# ------------------------------------------------------------------ données

def annotations() -> pd.DataFrame:
    parts = [pd.read_csv(f, dtype=str, keep_default_na=False)
             for f in sorted((TRAITEMENT / "annotations").glob("niveau_*.csv"))]
    a = pd.concat(parts, ignore_index=True)
    a = a[a.niveau_px != ""].sort_values("annote_le").drop_duplicates("fichier", keep="last")
    a["niveau_px"] = a.niveau_px.astype(float)
    split = pd.read_csv(TRAITEMENT / "split_train_test.csv")
    return a[["fichier", "niveau_px"]].merge(split, on="fichier").sort_values("fichier").reset_index(drop=True)


def metriques(err: np.ndarray) -> dict:
    a = np.abs(err)
    return {"erreur moyenne": a.mean(), "erreur médiane": np.median(a), "erreur max": a.max(),
            "biais moyen": err.mean(),
            "à 5 px près": 100 * (a <= 5).mean(), "à 10 px près": 100 * (a <= 10).mean(),
            "à 20 px près": 100 * (a <= 20).mean()}


# ------------------------------------------------------------------ tableau

def fr(x: float, d: int = 1) -> str:
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def tableau(res: dict, naif: float, n: int):
    lignes = ["erreur moyenne", "erreur médiane", "erreur max", "biais moyen*",
              "à 5 px près", "à 10 px près", "à 20 px près", "vérité dans la zone",
              "zone de recherche", "durée"]
    cles = list(METHODES)
    entetes = [f"Test ({n} photos)"] + [METHODES[k][0] for k in cles] + ["DINOv2 ViT-S\n(rappel)", "Référence\nnaïve**"]

    def cellule(k, ligne):
        r = res[k]
        m = r["m"]
        return {
            "erreur moyenne": f"{fr(m['erreur moyenne'])} px",
            "erreur médiane": f"{fr(m['erreur médiane'])} px",
            "erreur max": f"{fr(m['erreur max'], 0)} px",
            "biais moyen*": f"{'+' if m['biais moyen'] > 0 else ''}{fr(m['biais moyen'])} px",
            "à 5 px près": f"{fr(m['à 5 px près'])} %",
            "à 10 px près": f"{fr(m['à 10 px près'])} %",
            "à 20 px près": f"{fr(m['à 20 px près'])} %",
            "vérité dans la zone": f"{fr(r['dans_zone'])} %",
            "zone de recherche": f"{r['zone'][0]}–{r['zone'][1]} px",
            "durée": f"{fr(r['duree'], 0)} s\n(CPU Mac, 10 cœurs)",
        }[ligne]

    # meilleure valeur parmi les 3 méthodes (sens selon la ligne)
    valeur = {"erreur moyenne": ("erreur moyenne", min), "erreur médiane": ("erreur médiane", min),
              "erreur max": ("erreur max", min), "biais moyen*": ("biais moyen", lambda v: min(v, key=abs)),
              "à 5 px près": ("à 5 px près", max), "à 10 px près": ("à 10 px près", max),
              "à 20 px près": ("à 20 px près", max)}
    texte, gras = [], []
    for ligne in lignes:
        rangee = [ligne] + [cellule(k, ligne) for k in cles]
        g = [False] * (len(cles) + 1)
        if ligne in valeur:
            cle_m, choix = valeur[ligne]
            vals = [res[k]["m"][cle_m] for k in cles]
            best = choix(vals)
            g =[False] + [v == best for v in vals]
        elif ligne == "durée":
            d = [res[k]["duree"] for k in cles]
            g = [False] + [v == min(d) for v in d]
        base = ligne.rstrip("*")
        if base in DINOV2:
            rangee.append(f"{fr(DINOV2[base])} {'%' if base.startswith('à') else 'px'}")
        else:
            rangee.append("9 min\n(RTX 3080 Ti)" if ligne == "durée" else "—")
        rangee.append(f"{fr(naif)} px" if ligne == "erreur moyenne" else "—")
        texte.append(rangee)
        gras.append(g + [False, False])

    fig = plt.figure(figsize=(20, 7.9))
    ax = fig.add_axes([0.008, 0.0, 0.984, 1.0])
    ax.axis("off")
    t = ax.table(cellText=texte, colLabels=entetes, cellLoc="center", loc="upper center",
                 bbox=[0, 0.12, 1, 0.80])
    t.auto_set_font_size(False)
    bleu, gris, creme, bord = "#e7edf7", "#f4f4f4", "#fff4e5", "#c8c8c8"
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(bord)
        c.set_text_props(fontsize=14)
        if i == 0:
            c.set_facecolor(bleu)
            c.set_text_props(fontweight="bold", fontsize=14.5)
        elif j == 0:
            c.set_facecolor(gris)
        elif 1 <= j <= len(cles):
            c.set_facecolor(creme)
            if gras[i - 1][j]:
                c.set_text_props(fontweight="bold")
        else:
            c.set_facecolor("white")
            if j == len(cles) + 1:
                c.set_text_props(color="#666666")
    fig.text(0.008, 0.965, f"Hauteur d'eau sur les {n} photos test : méthodes « lignes de pixels » (sans apprentissage)",
             fontsize=19, va="top")
    fig.text(0.008, 0.055,
             "* biais = prédiction − vérité (positif : ligne prédite trop bas).   "
             "** référence naïve : prédire toujours la hauteur moyenne du train.   "
             "En gras : meilleure des 3 méthodes.   « vérité dans la zone » : part des photos dont le niveau annoté "
             "tombe dans la zone de recherche fixe de la méthode (hors zone, l'erreur est inévitable).",
             fontsize=11.5, color="#444444", wrap=True)
    SORTIE_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(SORTIE_PNG, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    df = annotations()
    test = df[df.split == "test"].reset_index(drop=True)
    naif = float(np.abs(test.niveau_px - df.loc[df.split == "train", "niveau_px"].mean()).mean())
    n = len(test)
    print(f"{n} photos test annotées ; référence naïve : {naif:.1f} px")

    res, preds = {}, test[["fichier", "niveau_px"]].copy()
    with Pool() as pool:
        for k, (nom, _, module) in METHODES.items():
            t0 = time.time()
            y = np.array(pool.map(_un, [(k, f) for f in test.fichier], chunksize=8), dtype=float)
            duree = time.time() - t0
            zone = (int(HAUTEUR * module.Y_MIN_RATIO), int(HAUTEUR * module.Y_MAX_RATIO))
            v = test.niveau_px.to_numpy()
            res[k] = {"m": metriques(y - v), "duree": duree, "zone": zone,
                      "dans_zone": 100 * ((v >= zone[0]) & (v < zone[1])).mean()}
            preds[f"pred_{k}"] = y.astype(int)
            m = res[k]["m"]
            print(f"{nom.splitlines()[0]:18s} moy {m['erreur moyenne']:6.1f}  méd {m['erreur médiane']:6.1f}  "
                  f"max {m['erreur max']:6.0f}  biais {m['biais moyen']:+6.1f}  "
                  f"5px {m['à 5 px près']:5.1f}%  10px {m['à 10 px près']:5.1f}%  20px {m['à 20 px près']:5.1f}%  "
                  f"zone {res[k]['dans_zone']:5.1f}%  ({duree:.0f} s)")

    SORTIE_CSV.parent.mkdir(exist_ok=True)
    preds.to_csv(SORTIE_CSV, index=False)
    tableau(res, naif, n)
    print(f"-> {SORTIE_CSV.relative_to(ROOT)}\n-> {SORTIE_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
