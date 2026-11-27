"""Lignes de pixels (sans apprentissage) : Stats RGB, Canny + gradient, Texture (variance locale).

Détecteurs de Tristan (Lignes_pixels.py, Lignes_pixels_canny_gradient.py, Lignes_pixels_texture.py),
appliqués comme dans evaluation_lignes_pixels.py, mais une photo à la fois (un seul cœur) pour mesurer
le temps par image : lecture du JPEG + détection.

    .venv/bin/python comparaison/methode_lignes.py
"""
from __future__ import annotations

import cv2

from commun import ROOT, Chrono, donnees, enregistrer, nom_cpu

import Lignes_pixels as rgb                          # noqa: E402  (racine du projet, voir commun.py)
import Lignes_pixels_canny_gradient as canny         # noqa: E402
import Lignes_pixels_texture as texture              # noqa: E402

IMAGES = ROOT / "data" / "images"
CATEGORIE = "Lignes de pixels (sans apprentissage)"


def pred_rgb(chemin) -> int:
    return int(rgb.detect_waterline(chemin)[3])


def pred_canny(chemin) -> int:
    return int(canny.detect_waterline(chemin)[5])


def pred_texture(chemin) -> int:
    image = cv2.imread(str(chemin))
    h = image.shape[0]
    y_min, y_max = int(h * texture.Y_MIN_RATIO), int(h * texture.Y_MAX_RATIO)
    profil, _ = texture.compute_texture_profile(image)
    return int(texture.method_gradient(profil, y_min, y_max)[0])


METHODES = {"lignes_rgb": ("Stats RGB", pred_rgb), "lignes_canny": ("Canny + gradient", pred_canny),
            "lignes_texture": ("Texture (variance locale)", pred_texture)}


def main():
    cv2.setNumThreads(1)                             # une photo à la fois, un cœur : temps comparables
    test = donnees().query("split == 'test'").reset_index(drop=True)
    print(f"Lignes de pixels : {len(test)} photos test, CPU {nom_cpu()}")
    for cle, (nom, predire) in METHODES.items():
        chrono, y = Chrono(), []
        for f in test.fichier:
            with chrono:
                y.append(predire(IMAGES / f))
        p = test[["fichier", "niveau_px"]].assign(prediction_px=y)
        enregistrer(cle, CATEGORIE, nom, p, apprentissage="aucun", duree_entrainement_s=None,
                    temps_image_ms=chrono.mediane_ms(), machine_image="CPU 1 cœur",
                    duree_totale_test_s=chrono.total_s())


if __name__ == "__main__":
    main()
