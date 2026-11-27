"""Données pour l'encodeur : photos annotées (niveau_px), découpage train / val / test, augmentations.

- Annotations : data_treatment/annotations/niveau_*.csv (une ligne par photo annotée).
- Découpage : data_treatment/split_train_test.csv (semaines entières, voir split_train_test.py).
  La validation (pour choisir la meilleure époque) est prise DANS train : une semaine de train sur
  quatre, jamais dans test. Le test ne sert qu'à l'évaluation finale.
- Entrée du réseau : la bande y ∈ [Y0, Y1] de l'image (là où se trouve la frontière, 613-903 px
  dans les annotations), redimensionnée à TAILLE.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "images"
TRAITEMENT = ROOT / "data_treatment"

W0 = 1919                    # largeur des images
Y0, Y1 = 300, 1350           # bande donnée au réseau (px, image d'origine)
TAILLE = (952, 518)          # (largeur, hauteur) de l'entrée : multiples de 14 (patchs DINOv2)
DECALAGE_MAX = 40            # augmentation : la bande est décalée verticalement de ± 40 px
MOYENNE, ECART_TYPE = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # normalisation ImageNet


def charger_annotations() -> pd.DataFrame:
    """Photos annotées : fichier, niveau_px, annotateur, semaine, split (train / val / test), decalage_y."""
    parts = []
    for f in sorted((TRAITEMENT / "annotations").glob("niveau_*.csv")):
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        parts.append(d.assign(annotateur=f.stem.removeprefix("niveau_")))
    ann = pd.concat(parts, ignore_index=True)
    ann = ann[ann.niveau_px != ""].sort_values("annote_le").drop_duplicates("fichier", keep="last")
    ann["niveau_px"] = ann.niveau_px.astype(float)

    split = pd.read_csv(TRAITEMENT / "split_train_test.csv")
    cat = pd.read_csv(TRAITEMENT / "catalogue_images.csv", usecols=["fichier", "decalage_y"])
    df = ann[["fichier", "niveau_px", "annotateur"]].merge(split, on="fichier").merge(cat, on="fichier")

    # validation : une semaine de train (avec annotations) sur quatre
    semaines = sorted(df.loc[df.split == "train", "semaine"].unique())
    df.loc[df.semaine.isin(semaines[2::4]), "split"] = "val"
    return df.sort_values("fichier").reset_index(drop=True)


class BandeRiviere(Dataset):
    """Renvoie (image normalisée, y relatif dans la bande [0, 1], haut de la bande en px, nom du fichier)."""

    def __init__(self, df: pd.DataFrame, augmenter: bool = False):
        self.df = df.reset_index(drop=True)
        self.augmenter = augmenter
        self.couleur = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.03)
        self.normaliser = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MOYENNE, ECART_TYPE)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(IMAGES / r.fichier)
        im.draft("RGB", (im.size[0] // 2, im.size[1] // 2))     # décodage JPEG à 1/2 : plus rapide
        im = im.convert("RGB")
        s = W0 / im.size[0]

        # décalage vertical aléatoire de la bande (la caméra bouge) : le niveau relatif bouge avec
        haut = Y0 + (random.randint(-DECALAGE_MAX, DECALAGE_MAX) if self.augmenter else 0)
        im = im.crop((0, round(haut / s), im.size[0], round((haut + Y1 - Y0) / s))).resize(TAILLE, Image.BILINEAR)
        if self.augmenter:
            im = self.couleur(im)
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)          # la ligne reste à la même hauteur
        y_rel = (r.niveau_px - haut) / (Y1 - Y0)
        return self.normaliser(im), torch.tensor(y_rel, dtype=torch.float32), haut, r.fichier


def vers_px(y_rel: torch.Tensor | np.ndarray, haut) -> np.ndarray:
    """y relatif dans la bande -> y en px de l'image d'origine."""
    return np.asarray(haut, dtype=float) + np.asarray(y_rel, dtype=float) * (Y1 - Y0)
