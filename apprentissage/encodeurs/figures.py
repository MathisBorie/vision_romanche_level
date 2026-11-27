"""Figures des résultats sur le test, à partir de encodeur/resultats/<encodeur>/predictions_test.csv.

    .venv/bin/python encodeur/figures.py                      # exemples de DINOv2 + histogramme des 3 encodeurs
    .venv/bin/python encodeur/figures.py --encodeur resnet50  # exemples d'un autre encodeur

Produit :
    resultats/<encodeur>/exemples_test.png   photos test, ligne annotée (verte) et prédite (rouge, tirets),
                                             erreurs typiques (médiane) puis de plus en plus grandes, jusqu'aux pires (paires + / -)
    resultats/histogramme_erreurs.png        distribution de « prédite - vraie » pour chaque encodeur
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ICI = Path(__file__).resolve().parent
RESULTATS = ICI / "resultats"
IMAGES = ICI.parent / "data" / "images"

# encodeur -> (nom affiché, couleur). Couleurs fixes par encodeur (palette catégorielle validée).
ENCODEURS = {"dinov2": ("DINOv2 ViT-S (gelé)", "#2a78d6"),
             "convnext": ("ConvNeXt-Tiny", "#eb6834"),
             "resnet50": ("ResNet-50", "#1baf7a")}
ANNOTE, PREDIT = "#00e000", "#ff2020"      # lignes sur les photos : vives pour ressortir sur l'eau et la roche
ENCRE, ENCRE_2, GRILLE = "#0b0b0b", "#52514c", "#e1e0d9"
DEMI_HAUTEUR = 160                          # px de photo affichés au-dessus et au-dessous des lignes


def style_axes(ax):
    for c in ("top", "right"):
        ax.spines[c].set_visible(False)
    for c in ("left", "bottom"):
        ax.spines[c].set_color("#c3c2b7")
    ax.tick_params(colors=ENCRE_2, labelsize=9)
    ax.grid(axis="y", color=GRILLE, lw=0.8)
    ax.set_axisbelow(True)


def choisir_exemples(p: pd.DataFrame) -> pd.DataFrame:
    """Paires d'erreurs + / - de taille croissante : typique (médiane), 75e, 95e centile, puis les pires."""
    p = p.assign(abs=p.erreur_px.abs())
    choix = []
    for q in (0.5, 0.75, 0.95, 1.0):
        for signe in (1, -1):
            cote = p[np.sign(p.erreur_px) == signe]
            if cote.empty:
                continue
            cible = cote["abs"].quantile(q)
            choix.append(cote.loc[(cote["abs"] - cible).abs().idxmin()])
    return pd.DataFrame(choix).drop_duplicates("fichier")


def figure_exemples(encodeur: str):
    p = pd.read_csv(RESULTATS / encodeur / "predictions_test.csv")
    ex = choisir_exemples(p)
    fig, axes = plt.subplots(len(ex), 1, figsize=(10, 2.05 * len(ex)))
    for ax, r in zip(np.atleast_1d(axes), ex.itertuples()):
        im = Image.open(IMAGES / r.fichier).convert("RGB")
        centre = (r.niveau_px + r.prediction_px) / 2
        y0 = int(np.clip(centre - DEMI_HAUTEUR, 0, im.size[1] - 2 * DEMI_HAUTEUR))
        ax.imshow(np.asarray(im.crop((0, y0, im.size[0], y0 + 2 * DEMI_HAUTEUR))),
                  extent=(0, im.size[0], y0 + 2 * DEMI_HAUTEUR, y0))
        ax.axhline(r.niveau_px, color=ANNOTE, lw=2, label=f"annoté : {r.niveau_px:.0f} px")
        ax.axhline(r.prediction_px, color=PREDIT, lw=2, ls=(0, (6, 3)), label=f"prédit : {r.prediction_px:.0f} px")
        sens = "eau prédite trop basse" if r.erreur_px > 0 else "eau prédite trop haute"
        ax.set_title(f"erreur {r.erreur_px:+.0f} px  ({sens})   —   {r.fichier[:-4]}",
                     loc="left", fontsize=10, color=ENCRE)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        ax.axis("off")
    fig.suptitle(f"{ENCODEURS[encodeur][0]} — photos test : erreur typique (médiane), 75e et 95e centiles, "
                 "puis les pires", x=0.01, ha="left", fontsize=11, color=ENCRE)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    sortie = RESULTATS / encodeur / "exemples_test.png"
    fig.savefig(sortie, dpi=110, facecolor="white")
    plt.close(fig)
    return sortie


def figure_histogramme(borne: int = 30, pas: int = 2):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bords = np.arange(-borne - pas, borne + pas + 1, pas)
    for cle, (nom, couleur) in ENCODEURS.items():
        f = RESULTATS / cle / "predictions_test.csv"
        if not f.exists():
            continue
        e = pd.read_csv(f).erreur_px.to_numpy()
        e_coupe = np.clip(e, -borne - pas / 2, borne + pas / 2)        # au-delà de ± borne : dans la case du bord
        ax.hist(e_coupe, bins=bords, histtype="step", lw=2, color=couleur,
                label=f"{nom} : |erreur| médiane {np.median(np.abs(e)):.1f} px, moyenne {np.abs(e).mean():.1f} px")
        n = len(e)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.4)                      # place pour la légende au-dessus des barres
    ax.axvline(0, ymax=1 / 1.4, color=ENCRE, lw=1)               # s'arrête sous la légende
    style_axes(ax)
    ax.set_xlim(-borne - pas, borne + pas)
    ax.set_xlabel(f"prédite - vraie (px)   (bords = au-delà de ± {borne})   > 0 : eau prédite trop basse",
                  color=ENCRE_2, fontsize=9)
    ax.set_ylabel("photos test", color=ENCRE_2, fontsize=9)
    ax.set_title(f"Erreur sur le test ({n} photos, 3 semaines jamais vues)", loc="left", fontsize=11, color=ENCRE)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    sortie = RESULTATS / "histogramme_erreurs.png"
    fig.savefig(sortie, dpi=130, facecolor="white")
    plt.close(fig)
    return sortie


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encodeur", choices=list(ENCODEURS), default="dinov2")
    a = p.parse_args()
    print(figure_exemples(a.encodeur).relative_to(ICI.parent))
    print(figure_histogramme().relative_to(ICI.parent))


if __name__ == "__main__":
    main()
