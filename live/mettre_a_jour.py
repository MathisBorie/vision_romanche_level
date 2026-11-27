"""Image « en direct » de la page d'accueil : dernière photo de la caméra + niveau d'eau mesuré par DINOv2.

Lancé toutes les 15 min par .github/workflows/niveau_live.yml (GitHub Actions) :

    python live/mettre_a_jour.py --verifier SORTIE   # écrit nouvelle=true/false dans $GITHUB_OUTPUT
    python live/mettre_a_jour.py SORTIE              # télécharge, mesure, dessine

SORTIE est le dossier de la branche `live` (une seule image, écrasée à chaque fois) :
    niveau_live.jpg   photo annotée affichée en haut du README (échelle 1,0 / 1,5 / 1,8 m de Simon)
    niveau_live.json  dernière mesure (fichier du bucket, date, niveau)
    historique.csv    toutes les mesures depuis la mise en route

Identifiants du bucket : variables d'environnement S3_BUCKET, AWS_REGION, AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY (secrets GitHub du dépôt), comme bucket.py.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3

ROOT = Path(__file__).resolve().parent.parent
CAMERA = "romanche-01/"
HEURE_LOCALE = ZoneInfo("Europe/Paris")         # la caméra est à Livet (Isère)
MOTIF = re.compile(r"(\d{4})/(\d{2})/(\d{2})/(\d{6})_[^/]*\.jpe?g$", re.IGNORECASE)


def client():
    return boto3.client("s3", region_name=os.getenv("AWS_REGION") or None,
                        endpoint_url=os.getenv("S3_ENDPOINT_URL") or None)


def derniere_photo(s3, bucket: str) -> str | None:
    """Clé de la photo la plus récente (cherche dans le mois en cours puis les précédents)."""
    mois = datetime.now(timezone.utc).replace(day=1)
    for _ in range(12):
        prefixe = f"{CAMERA}{mois:%Y/%m}/"
        cles = [o["Key"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefixe)
                for o in page.get("Contents", []) if MOTIF.search(o["Key"])]
        if cles:
            return max(cles, key=lambda k: MOTIF.search(k).groups())
        mois = (mois - timedelta(days=1)).replace(day=1)
    return None


def date_prise(cle: str) -> datetime:
    a, m, j, hms = MOTIF.search(cle).groups()
    return datetime.strptime(f"{a}{m}{j}{hms}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def dessiner(photo, y: float, quand: datetime, metres: float, decalage: float, echelle):
    """Photo réduite, échelle de Simon, ligne du niveau mesuré et bandeau avec la date et la valeur."""
    from PIL import Image, ImageDraw, ImageFont

    def police(t):
        for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     # Ubuntu (GitHub Actions)
                  "/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(f, t)
            except OSError:
                pass
        return ImageFont.load_default(size=t)

    k = 960 / photo.width
    im = photo.convert("RGB").resize((960, round(photo.height * k)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    couleurs = {1.8: (60, 210, 110), 1.5: (245, 170, 40), 1.0: (240, 80, 80)}
    for y_ref, h in echelle:                                 # traits de Simon, recalés sur la caméra
        ye = (y_ref + decalage) * k
        for x in range(0, im.width - 70, 16):
            d.line([(x, ye), (x + 9, ye)], fill=couleurs.get(h, (200, 200, 200)), width=2)
        d.rounded_rectangle([im.width - 64, ye - 11, im.width - 6, ye + 11], 4, fill=(20, 24, 32))
        d.text((im.width - 35, ye), f"{h:.1f} m".replace(".", ","), fill=couleurs.get(h, (200, 200, 200)), anchor="mm",
               font=police(15))
    yy = y * k
    for e, c in ((6, (0, 0, 0)), (4, (60, 160, 255))):
        d.line([(0, yy), (im.width - 70, yy)], fill=c, width=e)


    local = quand.astimezone(HEURE_LOCALE)
    d.text((12, yy - 36), f"eau : {metres:.2f} m".replace(".", ","), font=police(26), fill=(255, 255, 255),
           stroke_width=3, stroke_fill=(0, 0, 0))
    bandeau = Image.new("RGB", (im.width, 46), (20, 24, 32))
    db = ImageDraw.Draw(bandeau)
    db.text((12, 10), f"Romanche à Livet — {local:%d/%m/%Y %H:%M} (heure de Paris)", font=police(22),
            fill=(255, 255, 255))
    db.text((im.width - 12, 13), "mesure DINOv2, échelle de Simon", font=police(16), fill=(170, 180, 200),
            anchor="ra")
    sortie = Image.new("RGB", (im.width, im.height + bandeau.height))
    sortie.paste(bandeau, (0, 0))
    sortie.paste(im, (0, bandeau.height))
    return sortie


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sortie", type=Path)
    ap.add_argument("--verifier", action="store_true", help="dit seulement s'il y a une photo nouvelle")
    ap.add_argument("--forcer", action="store_true", help="recalcule même si la photo n'a pas changé")
    a = ap.parse_args()
    a.sortie.mkdir(parents=True, exist_ok=True)
    etat_f = a.sortie / "niveau_live.json"
    etat = json.loads(etat_f.read_text()) if etat_f.exists() else {}

    s3, bucket = client(), os.environ["S3_BUCKET"]
    cle = derniere_photo(s3, bucket)
    if cle is None:
        sys.exit("aucune photo dans le bucket")
    nouvelle = a.forcer or cle != etat.get("cle")
    print(f"dernière photo : {cle} ({'nouvelle' if nouvelle else 'déjà traitée'})")
    if a.verifier:
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"nouvelle={'true' if nouvelle else 'false'}\n")
        return
    if not nouvelle:
        return

    from PIL import Image
    sys.path.insert(0, str(ROOT / "niveau_eau"))
    from niveau_eau import mesurer_niveau

    brut = s3.get_object(Bucket=bucket, Key=cle)["Body"].read()
    photo = Image.open(io.BytesIO(brut))
    import niveau_eau as ne
    y = mesurer_niveau(Image.open(io.BytesIO(brut)), methode=3)
    decalage = ne.decalage_camera(Image.open(io.BytesIO(brut)))
    if photo.size == (1920, 2560):                           # ancien mode portrait : même cadrage que le modèle
        photo, y = photo.crop((0, 807, 1919, 2560)), y - 807
        decalage = ne.decalage_camera(photo)
    metres = ne.en_metres_depuis_px(y - decalage)
    quand = date_prise(cle)
    dessiner(photo, y, quand, metres, decalage, ne.ECHELLE).save(a.sortie / "niveau_live.jpg", quality=85)

    maintenant = datetime.now(timezone.utc)
    etat = {"cle": cle, "prise_utc": quand.isoformat(), "hauteur_m": metres, "niveau_px": y,
            "decalage_camera_px": round(decalage, 1), "methode": "DINOv2",
            "mis_a_jour_utc": maintenant.isoformat(timespec="seconds")}
    etat_f.write_text(json.dumps(etat, indent=1, ensure_ascii=False))
    hist = a.sortie / "historique.csv"
    if not hist.exists():
        hist.write_text("prise_utc,hauteur_m,niveau_px,decalage_camera_px,cle\n")
    with hist.open("a") as f:
        f.write(f"{quand.isoformat()},{metres},{y},{decalage:.1f},{cle}\n")
    print(f"hauteur {metres:.2f} m, niveau {y:.1f} px, décalage caméra {decalage:+.1f} px ({quand.astimezone(HEURE_LOCALE):%d/%m %H:%M} à Paris)")


if __name__ == "__main__":
    main()
