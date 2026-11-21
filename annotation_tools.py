"""Outils pour l'annotation manuelle (par un modèle) de la frontière eau/berge.

- is_dark(img) : tri des images de nuit sans les montrer au modèle (0 token).
- make_view(img) : bande utile y 600-1000, compressée en largeur seulement
  (on garde la résolution verticale d'origine, c'est y qu'on mesure), avec grille.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

Y0, Y1 = 600, 1000                 # bande où se trouve la frontière
# Rognage large pour les traitements d'image : limites physiques de la scène, pas les niveaux déjà vus.
# Haut : sous le haut de l'enrochement, bien au-dessus du pied du mur (~670) que l'eau ne touche jamais.
# Bas : au-dessus de la plage de galets du premier plan (~1350), qui n'est pas la berge d'en face.
CROP_Y0, CROP_Y1 = 500, 1200
XS = list(range(100, 1919, 200))   # les 10 colonnes annotées
VIEW_W = 640                       # largeur de la vue (1919 -> 640, facteur 3)
DARK_MEAN = 20                     # luminosité moyenne (0-255) sous laquelle l'image est noire


def brightness(img: Image.Image) -> float:
    """Luminosité moyenne, calculée sur un décodage JPEG réduit (très rapide)."""
    img.draft("L", (240, 220))
    return float(np.asarray(img.convert("L"), dtype=float).mean())


def is_dark(img: Image.Image) -> bool:
    return brightness(img) < DARK_MEAN


VY0, VY1, VSY = 690, 910, 2       # vue zoomée : bande y 690-910 agrandie x2 verticalement


def make_view(img: Image.Image) -> Image.Image:
    """Vue d'annotation zoomée, pensée pour éviter que le modèle s'accroche à une ligne étiquetée :
    graduations discrètes et toutes identiques tous les 10 px, valeur écrite tous les 20 px
    dans les deux marges, colonnes rouges fines."""
    img = img.convert("RGB")
    sx = VIEW_W / img.size[0]
    band = img.crop((0, VY0, img.size[0], VY1)).resize((VIEW_W, (VY1 - VY0) * VSY), Image.LANCZOS)
    m, top = 30, 14                                       # marges (étiquettes y, noms de colonnes)
    view = Image.new("RGB", (VIEW_W + 2 * m, band.size[1] + top), (0, 0, 0))
    view.paste(band, (m, top))
    over = Image.new("RGBA", view.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for y in range(VY0, VY1 + 1, 10):
        yy = (y - VY0) * VSY + top
        d.line([(m, yy), (m + VIEW_W, yy)], fill=(255, 255, 0, 70), width=1)   # ligne pâle
        d.line([(m - 6, yy), (m, yy)], fill=(255, 255, 0, 255))                 # tick à gauche
        d.line([(m + VIEW_W, yy), (m + VIEW_W + 6, yy)], fill=(255, 255, 0, 255))
        if y % 20 == 0:
            d.text((1, yy - 5), str(y), fill=(255, 255, 0, 255))
            d.text((m + VIEW_W + 7, yy - 5), str(y), fill=(255, 255, 0, 255))
    for i, x in enumerate(XS):
        xx = m + x * sx
        d.line([(xx, top), (xx, view.size[1])], fill=(255, 0, 0, 200), width=1)
        d.text((min(xx - 6, m + VIEW_W - 16), 1), f"C{i + 1}", fill=(255, 0, 0, 255))
    return Image.alpha_composite(view.convert("RGBA"), over).convert("RGB")
