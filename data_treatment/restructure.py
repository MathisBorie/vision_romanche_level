"""Rassemble toutes les photos dans un seul dossier, renommées AAAA-MM-JJ-HHMMSS.jpg.

data/raw/romanche-01/2026/11/06/104512_02821fd2_s.jpg  ->  data/images/2026-11-06-104512.jpg

Par défaut, les fichiers sont liés (lien physique : pas d'espace disque en plus) et
data/raw/ reste intact, pour que bucket.py et les notebooks ne re-téléchargent rien.

Images portrait : les 1er, 3 et 4 juin 2026, la caméra était en mode portrait (1920 x 2560).
Même objectif, même échelle : une image paysage (1919 x 1753) correspond à la bande
y 807-2560, x 0-1919 de l'image portrait (mesuré par corrélation sur le mur). Ces images sont
recadrées vers ce format et écrites comme nouveaux fichiers (l'original dans data/raw/ n'est pas
modifié) ; le JPEG porte le commentaire "recadree-portrait" pour qu'on puisse les repérer.

Utilisation :
    .venv/bin/python data_treatment/restructure.py            # liens physiques (ou copie si impossible)
    .venv/bin/python data_treatment/restructure.py --move     # déplace les fichiers (vide data/raw/)
    .venv/bin/python data_treatment/restructure.py --src data/raw --dest data/images
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent   # racine du projet (data/)

# .../AAAA/MM/JJ/HHMMSS_xxxxxxxx_s.jpg
MOTIF = re.compile(r"(\d{4})/(\d{2})/(\d{2})/(\d{6})_[^/]*\.jpe?g$", re.IGNORECASE)

TAILLE_PORTRAIT = (1920, 2560)
RECADRAGE_PORTRAIT = (0, 807, 1919, 2560)   # (gauche, haut, droite, bas) -> 1919 x 1753
COMMENTAIRE_RECADRAGE = b"recadree-portrait"


def taille(chemin: Path) -> tuple[int, int]:
    with Image.open(chemin) as im:
        return im.size


def recadrer_portrait(src: Path, dest: Path):
    """Écrit dans dest la bande de src qui correspond au cadrage paysage (src n'est pas modifié)."""
    tmp = dest.with_suffix(".tmp")
    with Image.open(src) as im:
        exif = im.info.get("exif", b"")
        im.crop(RECADRAGE_PORTRAIT).save(tmp, "JPEG", quality=95, exif=exif, comment=COMMENTAIRE_RECADRAGE)
    os.replace(tmp, dest)


def nouveau_nom(chemin: Path) -> str | None:
    """Nom AAAA-MM-JJ-HHMMSS.jpg, ou None si le chemin ne suit pas l'arborescence attendue."""
    m = MOTIF.search(chemin.as_posix())
    return f"{m[1]}-{m[2]}-{m[3]}-{m[4]}.jpg" if m else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--dest", type=Path, default=ROOT / "data" / "images")
    p.add_argument("--move", action="store_true", help="déplacer au lieu de lier/copier")
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    faits = deja = ignores = recadrees = 0
    vus: dict[str, Path] = {}

    for src in sorted(args.src.rglob("*")):
        if not src.is_file():
            continue
        nom = nouveau_nom(src)
        if nom is None:                       # ex. _permission-test.jpg
            ignores += 1
            continue
        if nom in vus:                        # deux photos à la même seconde : ne devrait pas arriver
            raise RuntimeError(f"Doublon {nom} : {vus[nom]} et {src}")
        vus[nom] = src

        dest = args.dest / nom
        if dest.exists():
            if taille(dest) != TAILLE_PORTRAIT:
                deja += 1
                continue
            dest.unlink()                     # ancien lien vers un portrait : on le remplace par le recadrage

        if taille(src) == TAILLE_PORTRAIT:
            recadrer_portrait(src, dest)
            if args.move:
                src.unlink()
            recadrees += 1
        elif args.move:
            shutil.move(src, dest)
        else:
            try:
                os.link(src, dest)
            except OSError:                   # autre disque, système de fichiers sans liens...
                shutil.copy2(src, dest)
        faits += 1

    print(f"{faits} images {'déplacées' if args.move else 'ajoutées'} dans {args.dest}")
    print(f"dont {recadrees} images portrait recadrées en paysage")
    print(f"{deja} déjà présentes, {ignores} fichiers ignorés (nom non reconnu)")


if __name__ == "__main__":
    main()
