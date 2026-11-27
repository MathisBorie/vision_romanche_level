"""Image « en direct » de la page d'accueil : dernière photo de la caméra + niveau d'eau des 3 modèles.

Lancé toutes les 15 min par .github/workflows/niveau_live.yml (GitHub Actions) :

    python live/mettre_a_jour.py --verifier SORTIE   # écrit nouvelle=true/false dans $GITHUB_OUTPUT
    python live/mettre_a_jour.py SORTIE              # télécharge, mesure, dessine

SORTIE est le dossier de la branche `live` (une seule image, écrasée à chaque fois) :
    niveau_live.jpg   photo annotée affichée en haut du README (un trait par modèle, repères 1,0 / 1,5 / 1,8 m)
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


#: (méthode, nom court, couleur du trait), du plus précis au moins précis
MODELES = [(3, "DINOv2", (40, 220, 90)), (2, "SAM affiné", (40, 140, 255)), (1, "SAM brut", (255, 50, 50))]


def police(t):
    from PIL import ImageFont
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",          # Ubuntu (GitHub Actions)
              "/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, t)
        except OSError:
            pass
    return ImageFont.load_default(size=t)


def dessiner(photo, mesures: list[dict], quand: datetime, decalage: float, echelle):
    """Photo réduite, repères en mètres, un trait par modèle, date en haut et hauteurs en bas."""
    from PIL import Image, ImageDraw
    k = 960 / photo.width
    im = photo.convert("RGB").resize((960, round(photo.height * k)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for y_ref, h in echelle:                                 # repères 1,0 / 1,5 / 1,8 m, recalés sur la caméra
        ye = (y_ref + decalage) * k
        for x in range(0, im.width - 62, 14):
            d.line([(x, ye), (x + 7, ye)], fill=(255, 255, 255), width=1)
        d.text((im.width - 8, ye), f"{h:.1f} m".replace(".", ","), fill=(255, 255, 255), anchor="rm",
               font=police(15), stroke_width=2, stroke_fill=(0, 0, 0))
    for m in reversed(mesures):                              # le plus précis dessiné en dernier (au-dessus)
        yy = m["niveau_px"] * k
        d.line([(0, yy), (im.width - 62, yy)], fill=(0, 0, 0), width=5)
        d.line([(0, yy), (im.width - 62, yy)], fill=m["couleur"], width=3)

    fond = (20, 24, 32)
    haut = Image.new("RGB", (im.width, 40), fond)
    ImageDraw.Draw(haut).text((12, 20), f"Romanche à Livet — {quand.astimezone(HEURE_LOCALE):%d/%m/%Y %H:%M}",
                              font=police(21), fill=(255, 255, 255), anchor="lm")
    bas = Image.new("RGB", (im.width, 44), fond)
    db = ImageDraw.Draw(bas)
    x = 12
    for m in mesures:
        db.rectangle([x, 17, x + 22, 23], fill=m["couleur"])
        texte = f"{m['nom']} : {m['hauteur_m']:.2f} m".replace(".", ",")
        db.text((x + 30, 22), texte, font=police(19), fill=(255, 255, 255), anchor="lm")
        x += 30 + db.textlength(texte, font=police(19)) + 34
    sortie = Image.new("RGB", (im.width, haut.height + im.height + bas.height))
    sortie.paste(haut, (0, 0))
    sortie.paste(im, (0, haut.height))
    sortie.paste(bas, (0, haut.height + im.height))
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
    import niveau_eau as ne

    brut = s3.get_object(Bucket=bucket, Key=cle)["Body"].read()
    ouvrir = lambda: Image.open(io.BytesIO(brut))
    photo, retrait = ouvrir(), 0
    if photo.size == (1920, 2560):                           # ancien mode portrait : même cadrage que le modèle
        photo, retrait = photo.convert("RGB").crop((0, 807, 1919, 2560)), 807
    decalage = ne.decalage_camera(photo if retrait else ouvrir())
    mesures = []
    for methode, nom, couleur in MODELES:
        y = ne.mesurer_niveau(ouvrir(), methode=methode) - retrait
        mesures.append({"methode": methode, "nom": nom, "couleur": couleur, "niveau_px": y,
                        "hauteur_m": ne.en_metres_depuis_px(y - decalage)})
        print(f"  {nom:11s} {mesures[-1]['hauteur_m']:.2f} m  ({y:.1f} px)", flush=True)
    quand = date_prise(cle)
    dessiner(photo, mesures, quand, decalage, ne.ECHELLE).save(a.sortie / "niveau_live.jpg", quality=85)

    etat = {"cle": cle, "prise_utc": quand.isoformat(), "decalage_camera_px": round(decalage, 1),
            "mesures": {m["nom"]: {"hauteur_m": m["hauteur_m"], "niveau_px": m["niveau_px"]} for m in mesures},
            "mis_a_jour_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    etat_f.write_text(json.dumps(etat, indent=1, ensure_ascii=False))
    hist = a.sortie / "historique.csv"
    entete = "prise_utc,hauteur_dinov2_m,hauteur_sam_affine_m,hauteur_sam_brut_m,decalage_camera_px,cle\n"
    if not hist.exists() or hist.read_text().splitlines()[0] + "\n" != entete:
        hist.write_text(entete)                              # nouveau format : on repart de zéro
    with hist.open("a") as f:
        f.write(f"{quand.isoformat()},{','.join(str(m['hauteur_m']) for m in mesures)},{decalage:.1f},{cle}\n")
    print(f"photo du {quand.astimezone(HEURE_LOCALE):%d/%m %H:%M} (Paris), décalage caméra {decalage:+.1f} px")


if __name__ == "__main__":
    main()
