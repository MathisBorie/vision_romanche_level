"""Figure de synthèse : SAM 2 zero-shot (sam_bordure.py) vs SAM 2 à décodeur affiné (sam_affiner.py).

    .venv/bin/python figure_sam.py                          # photo par défaut
    .venv/bin/python figure_sam.py 2026-06-22-093013.jpg    # autre photo de data/images

Ligne du haut : image originale | masque + bordure zero-shot | masque + bordure décodeur affiné
Ligne du bas  : zoom sur la frontière | bordure colonne par colonne | erreurs sur les 883 photos test

Nécessite evaluation/sam_affine/decodeur.pt (produit par sam_affiner.py).
Sortie : figures/sam_zero_shot_vs_affine.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import sam_affiner as aff
import sam_bordure as zs

ROOT = Path(__file__).resolve().parent   # dossier apprentissage/sam
IMAGES = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir()) / "data" / "images"
TEST_ZS = ROOT / "evaluation" / "sam_test.csv"
TEST_AFF = ROOT / "evaluation" / "sam_affine" / "predictions_test.csv"
DECODEUR = ROOT / "evaluation" / "sam_affine" / "decodeur.pt"
SORTIE = ROOT / "figures" / "sam_zero_shot_vs_affine.png"
FICHIER_DEFAUT = "2026-06-22-083020.jpg"

C_ZS, C_AFF, C_VRAI = "#d62728", "#1f77b4", "#2ca02c"
ZOOM = 150                                           # demi-hauteur du zoom autour de la vérité (px)


def verite(fichier: str) -> float | None:
    for csv in (TEST_AFF, TEST_ZS):
        if csv.exists():
            d = pd.read_csv(csv)
            v = d.loc[d.fichier == fichier, "niveau_px"]
            if len(v):
                return float(v.iloc[0])
    return None


def zero_shot(img: Image.Image, device: str):
    seg = zs.Segmenteur(device)
    masque, score = seg.masque_eau(img)
    y_col = zs.bordure(masque)
    return masque, y_col, float(np.nanmedian(y_col)), score


@torch.no_grad()
def affine(fichier: str, device: str):
    """Même chemin que sam_affiner.evaluer : entrée 1024², prompt fixe, masque 256², ligne douce."""
    from transformers import Sam2Model
    model = Sam2Model.from_pretrained(zs.MODELE).to(device).eval()
    model.mask_decoder.load_state_dict(torch.load(DECODEUR, map_location="cpu"))
    pixels, _, _ = aff.Photos(pd.DataFrame({"fichier": [fichier], "niveau_px": [0.0]}))[0]
    pixels = pixels[None].to(device)
    emb = model.get_image_embeddings(pixels)
    pts, lab, boite = aff.prompts(1, device)
    out = model(image_embeddings=emb, input_points=pts, input_labels=lab, input_boxes=boite,
                multimask_output=False)
    logits = out.pred_masks[:, 0, 0].float()                                   # (1, 256, 256)
    y = float(aff.niveau_doux(logits)[0])
    # ligne douce colonne par colonne (niveau_doux fait la moyenne de ces valeurs)
    log_non_eau = F.logsigmoid(-aff.NETTETE * logits[:, aff.R0:aff.R1, :])
    y_col = aff.R0 * aff.H_RANG + torch.exp(torch.cumsum(log_non_eau, 1)).sum(1)[0] * aff.H_RANG
    proba = torch.sigmoid(aff.NETTETE * logits)[None]                          # pour l'affichage
    proba = F.interpolate(proba, size=(aff.H0, aff.W0), mode="bilinear", align_corners=False)[0, 0]
    x_col = (np.arange(aff.MASQUE) + 0.5) * aff.W0 / aff.MASQUE
    return proba.cpu().numpy(), x_col, y_col.cpu().numpy(), y


def superposer(ax, img, proba, couleur):
    ax.imshow(img)
    rgba = np.zeros((*proba.shape, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(couleur)
    rgba[..., 3] = 0.35 * proba
    ax.imshow(rgba)


def main():
    fichier = sys.argv[1] if len(sys.argv) > 1 else FICHIER_DEFAUT
    img = Image.open(IMAGES / fichier).convert("RGB")
    W, H = img.size
    y_vrai = verite(fichier)
    device = zs.choisir_device()
    print(f"{fichier} sur {device} ...")

    masque_zs, yc_zs, y_zs, score_zs = zero_shot(img, device)
    proba_aff, x_aff, yc_aff, y_aff = affine(fichier, device)
    ecart = lambda y: f" ({y - y_vrai:+.0f} px)" if y_vrai is not None else ""

    fig, axes = plt.subplots(2, 3, figsize=(22, 13.5), constrained_layout=True)

    # ---------------- ligne du haut
    ax = axes[0, 0]
    ax.imshow(img)
    ax.scatter(zs.X_POINTS, [zs.Y_EAU] * len(zs.X_POINTS), c="lime", s=40, edgecolors="k", label="prompt « eau »")
    ax.scatter(zs.X_POINTS, [zs.Y_BERGE] * len(zs.X_POINTS), c="magenta", s=40, edgecolors="k", label="prompt « berge »")
    x0, y0, x1, y1 = zs.BOITE
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="gold", lw=2, ls="--", label="boîte"))
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_title("Image originale + prompt fixe (identique pour toutes les photos)", fontsize=13)

    ax = axes[0, 1]
    superposer(ax, img, masque_zs.astype(float), C_ZS)
    ax.plot(np.arange(len(yc_zs)), yc_zs, color=C_ZS, lw=2)
    ax.set_title(f"SAM 2 zero-shot — masque + bordure\nmédiane y = {y_zs:.0f} px{ecart(y_zs)}  (score IoU {score_zs:.2f})",
                 fontsize=13)

    ax = axes[0, 2]
    superposer(ax, img, proba_aff, C_AFF)
    ax.plot(x_aff, yc_aff, color=C_AFF, lw=2)
    ax.set_title(f"SAM 2 décodeur affiné — masque + ligne douce\nmoyenne y = {y_aff:.0f} px{ecart(y_aff)}",
                 fontsize=13)
    for ax in axes[0]:
        ax.axis("off")

    # ---------------- ligne du bas
    centre = y_vrai if y_vrai is not None else (y_zs + y_aff) / 2
    haut, bas = max(0, int(centre - ZOOM)), min(H, int(centre + ZOOM))
    ax = axes[1, 0]
    ax.imshow(np.asarray(img)[haut:bas], extent=(0, W, bas, haut), aspect="auto")
    ax.plot(np.arange(len(yc_zs)), yc_zs, color=C_ZS, lw=2, label=f"zero-shot : {y_zs:.0f} px{ecart(y_zs)}")
    ax.plot(x_aff, yc_aff, color=C_AFF, lw=2, label=f"affiné : {y_aff:.0f} px{ecart(y_aff)}")
    if y_vrai is not None:
        ax.axhline(y_vrai, color=C_VRAI, lw=2.5, ls="--", label=f"vérité : {y_vrai:.0f} px")
    ax.set_ylim(bas, haut)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_title(f"Zoom sur la frontière (lignes {haut}–{bas})", fontsize=13)
    ax.set_xticks([])

    ax = axes[1, 1]
    ax.plot(np.arange(len(yc_zs)), yc_zs, color=C_ZS, lw=1.6, label="zero-shot (bord du masque, médiane glissante)")
    ax.plot(x_aff, yc_aff, color=C_AFF, lw=1.6, label="affiné (position attendue de la 1re eau)")
    ax.axhline(y_zs, color=C_ZS, ls=":", lw=1.4)
    ax.axhline(y_aff, color=C_AFF, ls=":", lw=1.4)
    if y_vrai is not None:
        ax.axhline(y_vrai, color=C_VRAI, lw=2, ls="--", label="vérité (une valeur par photo)")
    ax.set_ylim(bas, haut)
    ax.set_xlim(0, W)
    ax.set_xlabel("colonne (px)")
    ax.set_ylabel("ligne de l'image (px)")
    ax.set_title("Bordure colonne par colonne", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    ax = axes[1, 2]
    if TEST_ZS.exists() and TEST_AFF.exists():
        e_zs = pd.read_csv(TEST_ZS).ecart.dropna()
        e_aff = pd.read_csv(TEST_AFF).erreur_px.dropna()
        bins = np.arange(-60, 61, 2)
        for e, c, nom in [(e_zs, C_ZS, "zero-shot"), (e_aff, C_AFF, "affiné")]:
            ax.hist(e.clip(-60, 60), bins=bins, color=c, alpha=0.5,
                    label=f"{nom} : |err| moy. {e.abs().mean():.1f} px, ≤10 px {100 * (e.abs() <= 10).mean():.0f} %")
        ax.axvline(0, color=C_VRAI, lw=2, ls="--")
        ax.set_xlabel("erreur = prédiction − vérité (px, bornée à ±60)")
        ax.set_ylabel("nombre de photos")
        ax.set_title(f"Erreurs sur les photos test (n = {len(e_aff)})", fontsize=13)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
    else:
        ax.axis("off")

    fig.suptitle(f"{fichier} — SAM 2 (hiera-small) : zero-shot vs décodeur de masques affiné", fontsize=17)
    SORTIE.parent.mkdir(exist_ok=True)
    fig.savefig(SORTIE, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"Figure créée : {SORTIE}")


if __name__ == "__main__":
    main()
