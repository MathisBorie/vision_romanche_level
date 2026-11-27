"""Détection de la bordure eau / berge d'en face avec SAM 2, sur les images annotées à la main.

SAM 2 (facebook/sam2.1-hiera-small, via transformers) segmente l'eau à partir de prompts FIXES
(identiques pour toutes les images, l'annotation n'est jamais utilisée pour guider le modèle) :
    - points positifs « eau » sur une rangée en bas de l'image ;
    - points négatifs « berge / mur » en haut ;
    - une boîte qui exclut la végétation du haut.
La bordure est le bord supérieur de la plus grande composante connexe du masque d'eau, colonne par
colonne. y_sam = médiane de la bordure sur la largeur (même convention que niveau_px).

Utilisation :
    .venv/bin/python sam_bordure.py --n 5 --montrer          # 5 images au hasard + figures
    .venv/bin/python sam_bordure.py                          # toutes les images annotées (reprise auto)
    .venv/bin/python sam_bordure.py --annotateur mathis
    .venv/bin/python sam_bordure.py --fichier 2026-07-05-110011.jpg --montrer
    .venv/bin/python sam_bordure.py --liste evaluation/test_encodeur.csv --sortie evaluation/sam_test.csv
        (mêmes photos et même vérité que le test des encodeurs : comparaison directe)

Sorties :
    annotations/sam_bordure.csv   fichier, niveau_px, y_sam, ecart, score_sam, x100 ... x1900
    figures/sam/<fichier>.png     (avec --montrer) masque + bordure SAM (rouge) + niveau annoté (jaune)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "images"
ANNOTATIONS = ROOT / "data_treatment" / "annotations"
SORTIE = ROOT / "annotations" / "sam_bordure.csv"
FIGURES = ROOT / "figures" / "sam"

MODELE = "facebook/sam2.1-hiera-small"
XS = list(range(100, 2000, 200))                   # colonnes reportées, comme annotations/manuel.csv

# Prompts fixes (px, image d'origine 1919 x 1753)
Y_EAU = 1500                                       # rangée de points « eau »
Y_BERGE = 450                                      # rangée de points « berge / mur »
X_POINTS = list(range(200, 1800, 300))
BOITE = [0, 300, 1919, 1753]                       # x0, y0, x1, y1 : exclut la végétation du haut
LISSAGE = 31                                       # médiane glissante sur la bordure (colonnes)


def charger_annotations(annotateur: str | None) -> pd.DataFrame:
    morceaux = []
    for f in sorted(ANNOTATIONS.glob("niveau_*.csv")):
        nom = f.stem.removeprefix("niveau_")
        if annotateur and nom != annotateur:
            continue
        d = pd.read_csv(f)
        d["annotateur"] = nom
        morceaux.append(d)
    df = pd.concat(morceaux, ignore_index=True)
    df = df[(df["exploitable"] == "oui") & df["niveau_px"].notna()]
    df = df[df["fichier"].map(lambda f: (IMAGES / f).exists())]
    return df.drop_duplicates("fichier", keep="last").reset_index(drop=True)


def choisir_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Segmenteur:
    def __init__(self, device: str):
        from transformers import Sam2Model, Sam2Processor
        self.device = device
        self.processor = Sam2Processor.from_pretrained(MODELE)
        self.model = Sam2Model.from_pretrained(MODELE).to(device).eval()

    @torch.no_grad()
    def masque_eau(self, img: Image.Image) -> tuple[np.ndarray, float]:
        points = [[x, Y_EAU] for x in X_POINTS] + [[x, Y_BERGE] for x in X_POINTS]
        labels = [1] * len(X_POINTS) + [0] * len(X_POINTS)
        inputs = self.processor(
            images=img,
            input_points=[[points]],
            input_labels=[[labels]],
            input_boxes=[[BOITE]],
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**inputs, multimask_output=True)
        masques = self.processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"])[0]      # (objets, 3, H, W)
        scores = out.iou_scores.cpu()[0, 0]                          # (3,)
        k = int(scores.argmax())
        return masques[0, k].numpy().astype(bool), float(scores[k])


def bordure(masque: np.ndarray) -> np.ndarray:
    """y du bord supérieur de la plus grande composante d'eau, pour chaque colonne (NaN si absente).

    Garder toutes les composantes assez grandes (essayé) récupère l'eau calme morcelée, mais garde aussi
    des taches sur le mur : 31 photos de test empirent pour 1 corrigée. On garde donc la plus grande.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(masque.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.full(masque.shape[1], np.nan)
    plus_grande = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    eau = lab == plus_grande
    presente = eau.any(axis=0)
    y = np.where(presente, eau.argmax(axis=0), np.nan).astype(float)
    # médiane glissante (ignore les NaN)
    s = pd.Series(y).rolling(LISSAGE, center=True, min_periods=1).median()
    return s.to_numpy()


def figure(img: Image.Image, masque: np.ndarray, y: np.ndarray, niveau: float, y_sam: float, nom: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8.2))
    ax.imshow(img)
    couche = np.zeros((*masque.shape, 4))
    couche[masque] = (0.1, 0.5, 1.0, 0.35)
    ax.imshow(couche)
    ax.plot(np.arange(len(y)), y, color="red", lw=1.5, label=f"bordure SAM (médiane {y_sam:.0f})")
    ax.axhline(niveau, color="yellow", ls="--", lw=1.5, label=f"niveau annoté {niveau:.0f}")
    ax.scatter(X_POINTS, [Y_EAU] * len(X_POINTS), c="lime", s=15)
    ax.scatter(X_POINTS, [Y_BERGE] * len(X_POINTS), c="magenta", s=15)
    ax.set_title(f"{nom}  écart {y_sam - niveau:+.0f} px")
    ax.legend(loc="lower right")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{Path(nom).stem}.png", dpi=100)
    plt.close(fig)


def main():
    global SORTIE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, help="échantillon aléatoire de N images")
    ap.add_argument("--annotateur", help="seulement les annotations de cette personne")
    ap.add_argument("--fichier", nargs="+", help="seulement ces images")
    ap.add_argument("--montrer", action="store_true", help="enregistre une figure par image")
    ap.add_argument("--liste", type=Path, help="CSV (fichier, niveau_px) : ces photos, avec ce niveau comme vérité")
    ap.add_argument("--sortie", type=Path, default=SORTIE, help=f"CSV de sortie (défaut {SORTIE.relative_to(ROOT)})")
    ap.add_argument("--refaire", action="store_true", help="recalcule même les images déjà traitées")
    args = ap.parse_args()

    SORTIE = args.sortie
    if args.liste:
        df = pd.read_csv(args.liste)[["fichier", "niveau_px"]]
        df = df[df["fichier"].map(lambda f: (IMAGES / f).exists())].reset_index(drop=True)
    else:
        df = charger_annotations(args.annotateur)
    if args.fichier:
        df = df[df["fichier"].isin(args.fichier)]
    deja = pd.read_csv(SORTIE) if SORTIE.exists() else pd.DataFrame(columns=["fichier"])
    if not args.refaire:
        df = df[~df["fichier"].isin(deja["fichier"])]
    if args.n:
        df = df.sample(min(args.n, len(df)), random_state=0)
    print(f"{len(df)} images à traiter ({len(deja)} déjà dans {SORTIE.name})")
    if df.empty:
        resume(deja)
        return

    device = choisir_device()
    print(f"Chargement de {MODELE} sur {device} ...")
    seg = Segmenteur(device)

    lignes = []
    t0 = time.time()
    for i, r in enumerate(df.itertuples(), 1):
        img = Image.open(IMAGES / r.fichier).convert("RGB")
        masque, score = seg.masque_eau(img)
        y = bordure(masque)
        y_sam = float(np.nanmedian(y)) if np.isfinite(y).any() else np.nan
        ligne = {"fichier": r.fichier, "niveau_px": r.niveau_px, "y_sam": round(y_sam, 1),
                 "ecart": round(y_sam - r.niveau_px, 1), "score_sam": round(score, 3)}
        ligne.update({f"x{x}": (None if np.isnan(y[x]) else int(y[x])) for x in XS if x < len(y)})
        lignes.append(ligne)
        print(f"[{i}/{len(df)}] {r.fichier}  annoté {r.niveau_px:.0f}  SAM {y_sam:.0f}  "
              f"écart {ligne['ecart']:+.0f}  score {score:.2f}")
        if args.montrer:
            figure(img, masque, y, r.niveau_px, y_sam, r.fichier)
        if i % 20 == 0:                              # sauvegarde régulière
            deja = enregistrer(deja, lignes)
            lignes = []
    deja = enregistrer(deja, lignes)
    print(f"durée : {(time.time() - t0) / 60:.1f} min ({(time.time() - t0) / len(df):.2f} s/image, {device})")
    resume(deja)


def enregistrer(deja: pd.DataFrame, lignes: list[dict]) -> pd.DataFrame:
    if lignes:
        nouv = pd.DataFrame(lignes)
        deja = pd.concat([deja[~deja["fichier"].isin(nouv["fichier"])], nouv], ignore_index=True)
        SORTIE.parent.mkdir(parents=True, exist_ok=True)
        deja.sort_values("fichier").to_csv(SORTIE, index=False)
    return deja


def metriques(ecart: pd.Series) -> dict:
    """Mêmes métriques que encodeur/entrainer.py (erreur absolue en px)."""
    e = ecart.abs()
    return {"n": len(e), "erreur_moyenne_px": round(e.mean(), 2), "erreur_mediane_px": round(e.median(), 2),
            "erreur_max_px": round(e.max(), 1), "a_5px_pct": round(100 * (e <= 5).mean(), 1),
            "a_10px_pct": round(100 * (e <= 10).mean(), 1), "a_20px_pct": round(100 * (e <= 20).mean(), 1),
            "biais_moyen_px": round(ecart.mean(), 2)}


def resume(d: pd.DataFrame):
    if d.empty or "ecart" not in d:
        return
    manquants = d["ecart"].isna().sum()
    print(f"\n{len(d)} images" + (f" ({manquants} sans eau détectée, exclues)" if manquants else ""))
    for k, v in metriques(d["ecart"].dropna()).items():
        print(f"  {k:20s} {v}")


if __name__ == "__main__":
    main()
