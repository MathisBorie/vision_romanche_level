"""Figure de synthèse des trois détecteurs « lignes de pixels » sur une même photo.

    .venv/bin/python figure_methodes_lignes_pixels.py                         # photo par défaut
    .venv/bin/python figure_methodes_lignes_pixels.py 2026-06-22-093013.jpg   # autre photo de data/images

Ligne du haut  : image originale | stats RGB | contours Canny | carte de texture
Ligne du bas   : les trois lignes détectées (+ vérité) | signal de chaque méthode

Sortie : figures/methodes_lignes_pixels.png
"""
from __future__ import annotations

import sys
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

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "images"
PREDICTIONS = ROOT / "evaluation" / "predictions_lignes_pixels.csv"
SORTIE = ROOT / "figures" / "methodes_lignes_pixels.png"
FICHIER_DEFAUT = "2026-06-22-083020.jpg"

COULEURS = {"rgb": "#d62728", "canny": "#1f77b4", "texture": "#ff7f0e", "verite": "#2ca02c"}


def verite(fichier: str) -> float | None:
    if not PREDICTIONS.exists():
        return None
    df = pd.read_csv(PREDICTIONS)
    ligne = df.loc[df.fichier == fichier, "niveau_px"]
    return float(ligne.iloc[0]) if len(ligne) else None


def tracer_signal(ax, signal, y, y_min, y_max, couleur, titre, y_vrai):
    """Signal 1D en fonction de la hauteur, orienté comme l'image (y vers le bas)."""
    lignes = np.arange(len(signal))
    ax.plot(signal, lignes, color=couleur, lw=1.4)
    ax.axhspan(y_min, y_max, color="gray", alpha=0.12, label="zone de recherche")
    ax.axhline(y, color=couleur, ls="--", lw=1.6, label=f"détecté : {y} px")
    if y_vrai is not None:
        ax.axhline(y_vrai, color=COULEURS["verite"], lw=1.6, label=f"vérité : {y_vrai:.0f} px")
    ax.set_ylim(len(signal), 0)
    ax.set_ylabel("ligne de l'image (px)")
    ax.set_title(titre, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")


def main():
    fichier = sys.argv[1] if len(sys.argv) > 1 else FICHIER_DEFAUT
    chemin = IMAGES / fichier
    bgr = cv2.imread(str(chemin))
    if bgr is None:
        raise FileNotFoundError(chemin)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h = image.shape[0]
    y_vrai = verite(fichier)

    # --- Stats RGB
    _, _, score_rgb, y_rgb = rgb.detect_waterline(chemin)
    zone_rgb = int(h * rgb.Y_MIN_RATIO), int(h * rgb.Y_MAX_RATIO)

    # --- Canny + gradient
    _, _, contours, _, grad_canny, y_canny, *zone_canny = canny.detect_waterline(chemin)

    # --- Texture (même pipeline que l'évaluation : y_final = y_grad)
    zone_tex = int(h * texture.Y_MIN_RATIO), int(h * texture.Y_MAX_RATIO)
    profil_tex, carte_tex = texture.compute_texture_profile(bgr)
    y_tex, grad_tex = texture.method_gradient(profil_tex, *zone_tex)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1]})

    # ---------------- ligne du haut : images
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image originale", fontsize=13)

    axes[0, 1].imshow(image)
    axes[0, 1].axhline(y_rgb, color=COULEURS["rgb"], lw=2.5)
    axes[0, 1].set_title(f"Stats RGB — y = {y_rgb} px", fontsize=13)

    axes[0, 2].imshow(contours, cmap="gray")
    axes[0, 2].axhline(y_canny, color=COULEURS["canny"], lw=2.5)
    axes[0, 2].set_title(f"Contours Canny (fermeture horiz.) — y = {y_canny} px", fontsize=13)

    axes[0, 3].imshow(carte_tex, cmap="inferno")
    axes[0, 3].axhline(y_tex, color="white", lw=2.5)
    axes[0, 3].set_title(f"Texture (écart-type local 21×21) — y = {y_tex} px", fontsize=13)

    for ax in axes[0]:
        ax.axis("off")

    # ---------------- ligne du bas : comparaison + signaux
    ax = axes[1, 0]
    ax.imshow(image)
    for cle, y, nom in [("rgb", y_rgb, "Stats RGB"), ("canny", y_canny, "Canny"), ("texture", y_tex, "Texture")]:
        ecart = f" ({y - y_vrai:+.0f})" if y_vrai is not None else ""
        ax.axhline(y, color=COULEURS[cle], lw=2, ls="--", label=f"{nom} : {y} px{ecart}")
    if y_vrai is not None:
        ax.axhline(y_vrai, color=COULEURS["verite"], lw=2.5, label=f"Vérité : {y_vrai:.0f} px")
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)
    ax.set_title("Comparaison des lignes détectées", fontsize=13)
    ax.axis("off")

    tracer_signal(axes[1, 1], score_rgb, y_rgb, *zone_rgb, COULEURS["rgb"],
                  "Variation des stats RGB entre lignes", y_vrai)
    axes[1, 1].set_xlabel("score (Δ moyenne + Δ écart-type)")

    tracer_signal(axes[1, 2], grad_canny, y_canny, *zone_canny, COULEURS["canny"],
                  "Chute de la densité de contours", y_vrai)
    axes[1, 2].set_xlabel("−d(densité)/dy")

    tracer_signal(axes[1, 3], grad_tex, y_tex, *zone_tex, COULEURS["texture"],
                  "Gradient du profil de texture", y_vrai)
    axes[1, 3].set_xlabel("|d(texture)/dy| (Savitzky-Golay)")

    fig.suptitle(f"{fichier} — détection de la ligne d'eau par « lignes de pixels »", fontsize=17)
    SORTIE.parent.mkdir(exist_ok=True)
    fig.savefig(SORTIE, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"Figure créée : {SORTIE}")


if __name__ == "__main__":
    main()
