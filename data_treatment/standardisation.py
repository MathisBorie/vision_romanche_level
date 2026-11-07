"""Mesure chaque photo de data/images/ et écrit les mesures dans catalogue_images.csv.

Étape 2 de la chaîne : restructure.py -> standardisation.py -> classification.py -> define_water_lvl.py
    .venv/bin/python data_treatment/standardisation.py

Ce script lit les images (quelques secondes pour ~5 500 images) et calcule tout ce qui se mesure
sur les pixels, en particulier le décalage de la caméra qui permet de ramener toutes les images
dans le même repère. Il ne décide rien : les classes (jour / nuit, exploitable...) sont posées
ensuite par classification.py à partir de ces mesures.

Colonnes écrites (recalculées à chaque lancement) :
    fichier, datetime, jour, heure   (heures en UTC)
    format          paysage (1919 x 1753) / recadree (portrait recadré par restructure.py) / autre
    luminosite      niveau de gris moyen de la bande utile (y 500-1200), 0-255
    contraste       écart-type du niveau de gris de la bande utile
    nettete         netteté de la zone fixe (mur + enrochement, en haut de l'image) :
                    variance du laplacien / luminosité. Ne dépend pas du niveau d'eau.
    pct_sature      % de pixels blancs saturés dans la bande utile (reflets du soleil, surexposition)
    decalage_x/y    déplacement de la scène par rapport à l'image de référence (px, pleine résolution),
                    mesuré sur la zone fixe par corrélation de phase. La caméra dérive (~20 px entre
                    juin et novembre) et bouge au cours de la journée avec le soleil (jusqu'à ~15 px) :
                    un niveau en pixels se compare d'une image à l'autre avec  niveau_px - decalage_y.
                    decalage_y > 0 : la scène est plus bas que sur la référence.
                    Vide si l'image est trop sombre pour être alignée.
    confiance_alignement   0-1, qualité de la mesure du décalage (< 0.2 : mesure non fiable)

Les autres colonnes du CSV (classes, annotations manuelles) sont conservées telles quelles.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent   # racine du projet (data/)
IMAGES = ROOT / "data" / "images"
CSV = Path(__file__).resolve().parent / "catalogue_images.csv"   # à côté des scripts

REFERENCE = "2026-11-03-104512.jpg"   # image nette, sert de référence pour le cadrage
S = 4                                 # calculs à 1/4 de la résolution (les seuils de classification.py en dépendent)
CROP_Y0, CROP_Y1 = 500, 1200          # bande utile, comme dans les notebooks
FIXE_Y1 = 800                         # zone fixe pour l'alignement : y 0-800 (mur + enrochement)
LUM_MIN_ALIGNEMENT = 20               # en dessous (nuit), pas d'alignement fiable

TAILLE_PAYSAGE = (1919, 1753)
COMMENTAIRE_RECADRAGE = b"recadree-portrait"   # posé par restructure.py

COLONNES = ["fichier", "datetime", "jour", "heure", "format", "luminosite", "contraste", "nettete",
            "pct_sature", "decalage_x", "decalage_y", "confiance_alignement"]


def _gris_reduit(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Image RGB et niveaux de gris décodées à ~1/S, et facteur d'échelle vers la pleine résolution."""
    im = Image.open(path)
    w0 = im.size[0]
    im.draft("RGB", (im.size[0] // S, im.size[1] // S))
    rgb = np.asarray(im.convert("RGB"))
    return rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), w0 / rgb.shape[1]


def _zone_fixe(gris: np.ndarray, s: float) -> np.ndarray:
    """Zone fixe, passe-haut (enlève l'éclairage) et fenêtre de Hann, pour la corrélation de phase."""
    z = gris[: round(FIXE_Y1 / s)]
    z = z - cv2.GaussianBlur(z, (0, 0), 8)
    return z * cv2.createHanningWindow(z.shape[::-1], cv2.CV_32F)


_REF = None


def _ref():
    global _REF
    if _REF is None:
        _, g, s = _gris_reduit(IMAGES / REFERENCE)
        _REF = _zone_fixe(g, s)
    return _REF


def mesurer(path: Path) -> dict:
    with Image.open(path) as im:
        taille, recadree = im.size, im.info.get("comment", b"") == COMMENTAIRE_RECADRAGE
    rgb, gris, s = _gris_reduit(path)
    y0, y1 = round(CROP_Y0 / s), round(CROP_Y1 / s)
    bande, bande_g = rgb[y0:y1], gris[y0:y1]
    fixe = gris[round(100 / s): round(700 / s)]

    d = dict(
        fichier=path.name,
        format="autre" if taille != TAILLE_PAYSAGE else "recadree" if recadree else "paysage",
        luminosite=bande_g.mean(),
        contraste=bande_g.std(),
        nettete=cv2.Laplacian(fixe, cv2.CV_32F).var() / (fixe.mean() + 1),
        pct_sature=100 * (bande.min(axis=2) >= 245).mean(),
        decalage_x=np.nan, decalage_y=np.nan, confiance_alignement=np.nan,
    )
    zone = _zone_fixe(gris, s)
    if d["format"] != "autre" and d["luminosite"] >= LUM_MIN_ALIGNEMENT and zone.shape == _ref().shape:
        (dx, dy), r = cv2.phaseCorrelate(_ref(), zone)
        d.update(decalage_x=dx * s, decalage_y=dy * s, confiance_alignement=r)
    return d


def mettre_en_forme(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df.fichier.str[:-4], format="%Y-%m-%d-%H%M%S")
    df["datetime"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["jour"] = dt.dt.strftime("%Y-%m-%d")
    df["heure"] = dt.dt.strftime("%H:%M")
    for c in ["luminosite", "contraste", "pct_sature", "decalage_x", "decalage_y"]:
        df[c] = df[c].round(1)
    df["nettete"] = df.nettete.round(2)
    df["confiance_alignement"] = df.confiance_alignement.round(2)
    return df[COLONNES]


def fusionner(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les autres colonnes du CSV existant (classes, annotations), sans les modifier."""
    if not CSV.exists():
        return df
    ancien = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    autres = [c for c in ancien.columns if c not in COLONNES]
    perdues = set(ancien.fichier) - set(df.fichier)
    if perdues:
        print(f"Attention : {len(perdues)} lignes du CSV n'ont plus d'image dans {IMAGES} (retirées)")
    df = df.merge(ancien[["fichier"] + autres], on="fichier", how="left")
    df[autres] = df[autres].fillna("")      # nouvelles images : classes et annotations vides
    return df


def main():
    fichiers = sorted(IMAGES.glob("*.jpg"))
    if not fichiers:
        raise FileNotFoundError(f"Aucune image dans {IMAGES} : lancer restructure.py d'abord")
    if not (IMAGES / REFERENCE).exists():
        raise FileNotFoundError(f"Image de référence absente : {IMAGES / REFERENCE}")

    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame(ex.map(mesurer, fichiers, chunksize=32))
    df = fusionner(mettre_en_forme(df))
    df.to_csv(CSV, index=False)

    print(f"{len(df)} images mesurées -> {CSV.name}")
    print(f"  format       {df.format.value_counts().to_dict()}")
    print(f"  alignées     {df.decalage_y.notna().sum()} (confiance médiane {df.confiance_alignement.median():.2f})")
    print("Lancer ensuite classification.py")


if __name__ == "__main__":
    main()
