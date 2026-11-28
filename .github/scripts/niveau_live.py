"""Image « en direct » de la page d'accueil : dernière photo de la caméra + niveau d'eau des 3 modèles.

Lancé toutes les 15 min par .github/workflows/niveau_live.yml (GitHub Actions) :

    python .github/scripts/niveau_live.py --verifier SORTIE   # écrit nouvelle=true/false dans $GITHUB_OUTPUT
    python .github/scripts/niveau_live.py SORTIE              # télécharge, mesure, dessine

SORTIE est le dossier de la branche `live` (une seule image, écrasée à chaque fois) :
    niveau_live.jpg   photo annotée affichée en haut du README (un trait par modèle, repères 1,0 / 1,5 / 1,8 m)
    niveau_live.json  dernière mesure (photo, date, niveau)
    historique.csv    toutes les mesures depuis la mise en route
    niveau_5j.png     graphe de la hauteur d'eau des 5 derniers jours (3 modèles)

À chaque passage, les photos des 5 derniers jours absentes de l'historique sont aussi mesurées
(au plus --rattrapage par passage, pour rester court).

Source des photos : l'API publique du site de la caméra (SOURCE ci-dessous), sans identifiants.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import urllib.request

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "niveau_eau").is_dir())
SOURCE = "https://romanche.simon-duchene.workers.dev"   # site de la caméra (photos toutes les 30 min, de jour)
HEURE_LOCALE = ZoneInfo("Europe/Paris")         # la caméra est à Livet (Isère)


def telecharger(chemin: str) -> bytes:
    req = urllib.request.Request(SOURCE + chemin, headers={"User-Agent": "niveau-romanche-live"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


JOURS = 5                                       # fenêtre du graphe et du rattrapage
LUMINOSITE_MIN = 60                             # photo de nuit (moyenne < 60 / 255) : pas de mesure
ENTETE = "prise_utc,hauteur_dinov2_m,hauteur_sam_affine_m,hauteur_sam_brut_m,decalage_camera_px,cle"


def _photo(it: dict) -> dict:
    prise = datetime.fromisoformat(it["captured_at"].replace("Z", "+00:00"))
    return {"cle": it["img"].rsplit("/", 1)[-1], "img": it["img"], "prise": prise}


def photos_recentes(depuis: datetime) -> list[dict]:
    """Photos publiées depuis `depuis`, de la plus récente à la plus ancienne : {"cle", "img", "prise"}."""
    photos, avant = [], None
    while True:
        d = json.loads(telecharger("/api/captures?limit=100" + (f"&before={avant}" if avant else "")))
        for it in d.get("items", []):
            p = _photo(it)
            if p["prise"] < depuis:
                return photos
            photos.append(p)
        avant = d.get("next_before")
        if not avant:
            return photos


def derniere_photo() -> dict | None:
    """Dernière photo publiée : {"cle", "img", "prise"} (cle = identifiant unique de la photo)."""
    items = json.loads(telecharger("/api/captures?limit=1")).get("items", [])
    return _photo(items[0]) if items else None


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
        ye = (y_ref + (decalage or 0)) * k
        for x in range(0, im.width - 62, 14):
            d.line([(x, ye), (x + 7, ye)], fill=(255, 255, 255), width=1)
        d.text((im.width - 8, ye), f"{h:.1f} m".replace(".", ","), fill=(255, 255, 255), anchor="rm",
               font=police(15), stroke_width=2, stroke_fill=(0, 0, 0))
    for m in reversed(mesures):                              # le plus précis dessiné en dernier (au-dessus)
        if m["niveau_px"] is None:
            continue
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
    if mesures[0]["hauteur_m"] is None:
        db.text((x, 22), "Photo de nuit : pas de mesure", font=police(19), fill=(255, 255, 255), anchor="lm")
    for m in mesures:
        if m["hauteur_m"] is None:
            continue
        db.rectangle([x, 17, x + 22, 23], fill=m["couleur"])
        texte = f"{m['nom']} : {m['hauteur_m']:.2f} m".replace(".", ",")
        db.text((x + 30, 22), texte, font=police(19), fill=(255, 255, 255), anchor="lm")
        x += 30 + db.textlength(texte, font=police(19)) + 34
    sortie = Image.new("RGB", (im.width, haut.height + im.height + bas.height))
    sortie.paste(haut, (0, 0))
    sortie.paste(im, (0, haut.height))
    sortie.paste(bas, (0, haut.height + im.height))
    return sortie


def mesurer(brut: bytes):
    """(photo dans le cadrage du modèle, mesures des 3 modèles, décalage caméra en px)."""
    from PIL import Image
    sys.path.insert(0, str(ROOT / "niveau_eau"))
    import niveau_eau as ne

    ouvrir = lambda: Image.open(io.BytesIO(brut))
    photo, retrait = ouvrir(), 0
    if photo.size == (1920, 2560):                           # ancien mode portrait : même cadrage que le modèle
        photo, retrait = photo.convert("RGB").crop((0, 807, 1919, 2560)), 807
    from PIL import ImageStat
    if ImageStat.Stat(photo.convert("L")).mean[0] < LUMINOSITE_MIN:   # nuit : les modèles répondent n'importe quoi
        return photo, [{"methode": me, "nom": n, "couleur": c, "niveau_px": None, "hauteur_m": None}
                       for me, n, c in MODELES], None
    decalage = ne.decalage_camera(photo if retrait else ouvrir())
    mesures = []
    for methode, nom, couleur in MODELES:
        y = ne.mesurer_niveau(ouvrir(), methode=methode) - retrait
        mesures.append({"methode": methode, "nom": nom, "couleur": couleur, "niveau_px": y,
                        "hauteur_m": ne.en_metres_depuis_px(y - decalage)})
    return photo, mesures, decalage


def lire_historique(hist: Path) -> dict[str, str]:
    """{clé de la photo: ligne CSV} (format courant uniquement, doublons retirés)."""
    if not hist.exists():
        return {}
    lignes = hist.read_text().splitlines()
    if not lignes or lignes[0] != ENTETE:
        return {}                                            # ancien format : on repart de zéro
    return {l.rsplit(",", 1)[-1]: l for l in lignes[1:] if l.strip()}


def ecrire_historique(hist: Path, lignes: dict[str, str]):
    tri = sorted(lignes.values(), key=lambda l: datetime.fromisoformat(l.split(",", 1)[0]))
    hist.write_text(ENTETE + "\n" + "".join(l + "\n" for l in tri))


def ligne_csv(quand: datetime, mesures: list[dict], decalage: float | None, cle: str) -> str:
    """Une ligne de historique.csv ; champs vides pour une photo de nuit."""
    h = ",".join("" if m["hauteur_m"] is None else str(m["hauteur_m"]) for m in mesures)
    return f"{quand.isoformat()},{h},{'' if decalage is None else f'{decalage:.1f}'},{cle}"


def graphe(hist: Path, fichier: Path, maintenant: datetime):
    """Hauteur d'eau des JOURS derniers jours, un trait par modèle (heure de Paris)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    debut = maintenant - timedelta(days=JOURS)
    points = []
    for l in lire_historique(hist).values():
        c = l.split(",")
        t = datetime.fromisoformat(c[0])
        if t >= debut and c[1]:                              # photos de nuit (champs vides) ignorées
            points.append((t.astimezone(HEURE_LOCALE).replace(tzinfo=None), *map(float, c[1:4])))
    points.sort()
    coupes = []                                              # trou de plus de 2 h (la nuit) : on coupe le trait
    for p in points:
        if coupes and p[0] - coupes[-1][0] > timedelta(hours=2):
            coupes.append((coupes[-1][0] + timedelta(hours=1), float("nan"), float("nan"), float("nan")))
        coupes.append(p)
    points = coupes
    fig, ax = plt.subplots(figsize=(9.6, 3.2), dpi=100)
    for i, (_, nom, couleur) in reversed(list(enumerate(MODELES))):
        t = [p[0] for p in points]
        h = [p[1 + i] for p in points]
        ax.plot(t, h, "-o", ms=2.5, lw=1.6, color=[c / 255 for c in couleur], label=nom, zorder=3 - i)
    ax.set_xlim(debut.astimezone(HEURE_LOCALE).replace(tzinfo=None),
                maintenant.astimezone(HEURE_LOCALE).replace(tzinfo=None))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.set_ylabel("hauteur d'eau (m)")
    ax.set_title(f"Hauteur d'eau — {JOURS} derniers jours", fontsize=11, loc="left")
    ax.grid(alpha=0.3)
    ax.grid(alpha=0.12, which="minor", axis="x")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower right", bbox_to_anchor=(1, 1), ncol=3, fontsize=9,
              frameon=False)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    fig.tight_layout()
    fig.savefig(fichier)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sortie", type=Path)
    ap.add_argument("--verifier", action="store_true", help="dit seulement s'il y a une photo nouvelle")
    ap.add_argument("--forcer", action="store_true", help="recalcule même si la photo n'a pas changé")
    ap.add_argument("--rattrapage", type=int, default=8,
                    help="nombre max de photos manquantes des derniers jours mesurées par passage")
    a = ap.parse_args()
    a.sortie.mkdir(parents=True, exist_ok=True)
    etat_f = a.sortie / "niveau_live.json"
    etat = json.loads(etat_f.read_text()) if etat_f.exists() else {}

    photo_api = derniere_photo()
    if photo_api is None:
        sys.exit("aucune photo publiée")
    cle = photo_api["cle"]
    nouvelle = a.forcer or cle != etat.get("cle")
    print(f"dernière photo : {cle} ({'nouvelle' if nouvelle else 'déjà traitée'})")
    if a.verifier:
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"nouvelle={'true' if nouvelle else 'false'}\n")
        return
    if not nouvelle:
        return

    sys.path.insert(0, str(ROOT / "niveau_eau"))
    import niveau_eau as ne

    photo, mesures, decalage = mesurer(telecharger(photo_api["img"]))
    for m in mesures:
        if m["hauteur_m"] is not None:
            print(f"  {m['nom']:11s} {m['hauteur_m']:.2f} m  ({m['niveau_px']:.1f} px)", flush=True)
    quand = photo_api["prise"]
    dessiner(photo, mesures, quand, decalage, ne.ECHELLE).save(a.sortie / "niveau_live.jpg", quality=85)

    etat = {"cle": cle, "prise_utc": quand.isoformat(),
            "decalage_camera_px": None if decalage is None else round(decalage, 1),
            "mesures": {m["nom"]: {"hauteur_m": m["hauteur_m"], "niveau_px": m["niveau_px"]} for m in mesures},
            "mis_a_jour_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    etat_f.write_text(json.dumps(etat, indent=1, ensure_ascii=False))
    hist = a.sortie / "historique.csv"
    lignes = lire_historique(hist)
    lignes[cle] = ligne_csv(quand, mesures, decalage, cle)

    maintenant = datetime.now(timezone.utc)
    manquantes = [p for p in photos_recentes(maintenant - timedelta(days=JOURS)) if p["cle"] not in lignes]
    for p in manquantes[:a.rattrapage]:                      # photos des derniers jours pas encore mesurées
        _, m, dec = mesurer(telecharger(p["img"]))
        lignes[p["cle"]] = ligne_csv(p["prise"], m, dec, p["cle"])
        print(f"  rattrapage {p['prise'].astimezone(HEURE_LOCALE):%d/%m %H:%M} : "
              + " / ".join("nuit" if x["hauteur_m"] is None else f"{x['hauteur_m']:.2f}" for x in m), flush=True)
    if len(manquantes) > a.rattrapage:
        print(f"  encore {len(manquantes) - a.rattrapage} photos à rattraper")
    ecrire_historique(hist, lignes)
    graphe(hist, a.sortie / "niveau_5j.png", maintenant)
    print(f"photo du {quand.astimezone(HEURE_LOCALE):%d/%m %H:%M} (Paris)")


if __name__ == "__main__":
    main()
