"""SAM 2.1-S (segmentation) : zéro-shot et décodeur affiné (scripts de Mathis, sam_bordure.py / sam_affiner.py).

    .venv/bin/python comparaison/methode_sam.py                 # zéro-shot puis affinage
    .venv/bin/python comparaison/methode_sam.py --zero-shot      # zéro-shot seulement
    .venv/bin/python comparaison/methode_sam.py --affine --epoques 4

- Zéro-shot : prompt fixe (points eau / berge + boîte), bord supérieur de la plus grande composante d'eau,
  médiane sur la largeur (sam_bordure.py). Aucun apprentissage.
- Affiné : seul le décodeur de masques apprend (sam_affiner.py), sur le même train / val que les encodeurs ;
  meilleure époque choisie sur val. Les sorties vont dans comparaison/resultats/sam_affine/ (les résultats
  Mac de evaluation/sam_affine/ ne sont pas touchés).
Temps par image : lecture de la photo + SAM + extraction de la ligne, une photo à la fois, sur GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from PIL import Image

from commun import RESULTATS, ROOT, Chrono, donnees, enregistrer, nom_cpu, nom_gpu

import sam_affiner                                   # noqa: E402  (racine du projet, voir commun.py)
import sam_bordure                                   # noqa: E402

IMAGES = ROOT / "data" / "images"
CATEGORIE = "SAM 2.1-S (segmentation)"


def synchro():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def zero_shot():
    test = donnees().query("split == 'test'").reset_index(drop=True)
    appareil = sam_bordure.choisir_device()
    print(f"SAM zéro-shot : {len(test)} photos test sur {appareil}")
    seg = sam_bordure.Segmenteur(appareil)
    seg.masque_eau(Image.open(IMAGES / test.fichier[0]).convert("RGB"))      # chauffe
    chrono, y = Chrono(), []
    for f in test.fichier:
        with chrono:
            masque, _ = seg.masque_eau(Image.open(IMAGES / f).convert("RGB"))
            b = sam_bordure.bordure(masque)
            y.append(float(np.nanmedian(b)) if np.isfinite(b).any() else np.nan)
    enregistrer("sam_zero_shot", CATEGORIE, "SAM zéro-shot", test[["fichier", "niveau_px"]].assign(prediction_px=y),
                apprentissage="aucun", duree_entrainement_s=None, temps_image_ms=chrono.mediane_ms(),
                machine_image=nom_gpu(), duree_totale_test_s=chrono.total_s())


def affine(epoques: int, n_temps: int, limite: int = 0):
    sortie = RESULTATS / "sam_affine"
    sortie.mkdir(parents=True, exist_ok=True)
    split = sortie / "split.csv"
    donnees().to_csv(split, index=False)              # même train / val / test que les encodeurs
    sam_affiner.SPLIT, sam_affiner.SORTIE = split, sortie
    sys.argv = ["sam_affiner.py", "--epoques", str(epoques)] + (["--limite", str(limite)] if limite else [])
    print(f"SAM décodeur affiné : {epoques} époques, sorties dans {sortie.relative_to(ROOT)}/")
    t0 = time.perf_counter()
    sam_affiner.main()
    duree = time.perf_counter() - t0
    m = json.loads((sortie / "metriques.json").read_text())

    # temps par image : photo -> SAM (encodeur + décodeur affiné) -> ligne, une à la fois
    from transformers import Sam2Model
    appareil = sam_bordure.choisir_device()
    model = Sam2Model.from_pretrained(sam_bordure.MODELE).to(appareil).eval()
    model.mask_decoder.load_state_dict(torch.load(sortie / "decodeur.pt"))
    test = donnees().query("split == 'test'").reset_index(drop=True)
    photos = sam_affiner.Photos(test.head(n_temps + 5))
    chrono = Chrono()
    with torch.no_grad():
        for i in range(len(photos)):
            synchro()
            t = time.perf_counter()
            pixels = photos[i][0][None].to(appareil)
            sam_affiner.predire(model, pixels).cpu()
            synchro()
            if i >= 5:
                chrono.durees.append(time.perf_counter() - t)
    import pandas as pd
    p = pd.read_csv(sortie / "predictions_test.csv")
    enregistrer("sam_affine", CATEGORIE, "SAM, décodeur affiné", p, apprentissage=f"{m['meilleure_epoque']} époques",
                duree_entrainement_s=duree, machine_entrainement=nom_gpu(), temps_image_ms=chrono.mediane_ms(),
                machine_image=nom_gpu(), epoques_lancees=epoques)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zero-shot", action="store_true", help="seulement le zéro-shot")
    ap.add_argument("--affine", action="store_true", help="seulement l'affinage")
    ap.add_argument("--epoques", type=int, default=4, help="époques d'affinage (défaut 4, comme le run Mac)")
    ap.add_argument("--n-temps", type=int, default=100, help="photos pour le temps par image de l'affiné")
    ap.add_argument("--limite", type=int, default=0, help="essai rapide : N photos par ensemble pour l'affinage")
    a = ap.parse_args()
    tout = not (a.zero_shot or a.affine)
    print(f"CPU {nom_cpu()} | GPU {nom_gpu()}")
    if tout or a.zero_shot:
        zero_shot()
    if tout or a.affine:
        affine(a.epoques, a.n_temps, a.limite)


if __name__ == "__main__":
    main()
