"""Tableau comparatif de toutes les méthodes sur les photos test, avec durée d'entraînement et temps par image.

    .venv/bin/python encodeur/tableau_resultats.py               # mesure les temps puis dessine le tableau
    .venv/bin/python encodeur/tableau_resultats.py --n 200       # temps mesuré sur 200 photos test
    .venv/bin/python encodeur/tableau_resultats.py --sans-cpu    # temps sur GPU seulement (plus rapide)

Pour chaque encodeur entraîné (encodeur/resultats/<encodeur>/modele.pt) :
- métriques du test (metriques.json) et biais moyen (predictions_test.csv) ;
- durée d'entraînement : somme des durées des époques (historique.csv, entraînement + validation) ;
- TEMPS PAR IMAGE, mesuré ici, une photo à la fois (lot de 1), après 10 photos de chauffe :
    préparation = lecture du JPEG, découpe et redimensionnement de la bande (CPU)
    réseau      = passage dans l'encodeur et la tête
  sur GPU (si disponible) et sur CPU. On garde la médiane sur --n photos test.
  Les temps sont enregistrés dans resultats/<encodeur>/temps_execution.json.

Les autres méthodes (lignes de pixels, POD, SAM) viennent de encodeur/autres_methodes.csv : valeurs mesurées
par l'équipe, à compléter à la main. Quand la durée est un calcul complet sur les photos test
(duree_totale_test_s), le temps par image en est déduit (durée / nombre de photos).

Sorties : resultats/tableau_resultats.png, resultats/tableau_resultats.csv
"""
from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from donnees import BandeRiviere, charger_annotations
from modele import DetecteurNiveau

ICI = Path(__file__).resolve().parent
RESULTATS = ICI / "resultats"
AUTRES = ICI / "autres_methodes.csv"
ENCODEURS = {"resnet50": "ResNet-50 (affiné)", "convnext": "ConvNeXt-Tiny (affiné)", "dinov2": "DINOv2 ViT-S (gelé)"}
CATEGORIE_ENC = "Encodeurs pré-entraînés"
CHAUFFE = 10


# ---------------------------------------------------------------- mesures
@torch.no_grad()
def temps_par_image(cle: str, photos: pd.DataFrame, appareil: torch.device) -> dict:
    """Médianes (ms) : préparation, réseau, total, une photo à la fois."""
    modele = DetecteurNiveau(cle)
    modele.load_state_dict(torch.load(RESULTATS / cle / "modele.pt", map_location="cpu")["modele"])
    modele.to(appareil).eval()
    ds = BandeRiviere(photos, augmenter=False)
    cuda = appareil.type == "cuda"
    prep, reseau = [], []
    for i in range(CHAUFFE + len(ds)):
        t0 = time.perf_counter()
        x = ds[i % len(ds)][0][None].to(appareil)
        if cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.autocast(appareil.type, dtype=torch.bfloat16, enabled=cuda):
            modele(x)
        if cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= CHAUFFE:
            prep.append(t1 - t0)
            reseau.append(t2 - t1)
    p, r = 1000 * np.median(prep), 1000 * np.median(reseau)
    return {"preparation_ms": round(p, 1), "reseau_ms": round(r, 1), "total_ms": round(p + r, 1), "n": len(ds)}


def nom_machine(appareil: str) -> str:
    if appareil == "cuda":
        return torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
    return f"CPU {torch.get_num_threads()} threads"


def lignes_encodeurs(n: int, avec_cpu: bool) -> list[dict]:
    photos = charger_annotations()
    photos = photos[photos.split == "test"].sample(frac=1, random_state=0).head(n)
    appareils = (["cuda"] if torch.cuda.is_available() else []) + (["cpu"] if avec_cpu or not torch.cuda.is_available() else [])
    lignes = []
    for cle, nom in ENCODEURS.items():
        dossier = RESULTATS / cle
        if not (dossier / "modele.pt").exists():
            print(f"{cle} : pas de modèle entraîné, ignoré")
            continue
        m = json.loads((dossier / "metriques.json").read_text())
        hist = pd.read_csv(dossier / "historique.csv")
        pred = pd.read_csv(dossier / "predictions_test.csv")
        temps = {}
        for a in appareils:
            print(f"{cle} : temps par image sur {a} ({len(photos)} photos)...", flush=True)
            temps[a] = temps_par_image(cle, photos, torch.device(a)) | {"machine": nom_machine(a)}
        (dossier / "temps_execution.json").write_text(json.dumps(temps, indent=2, ensure_ascii=False))
        t = m["test"]
        duree_min = hist.duree_s.sum() / 60
        machine_ent = nom_machine("cuda") if torch.cuda.is_available() else nom_machine("cpu")
        lignes.append({
            "categorie": CATEGORIE_ENC, "methode": nom, "erreur_moyenne": t["erreur_moyenne_px"],
            "erreur_mediane": t["erreur_mediane_px"], "erreur_max": t["erreur_max_px"],
            "biais": round(pred.erreur_px.mean(), 1), "a5": t["a_5px_pct"], "a10": t["a_10px_pct"],
            "a20": t["a_20px_pct"], "apprentissage": f"{m['meilleure_epoque']} époques",
            "duree": f"{duree_min:.0f} min ({machine_ent})",
            "temps_image": " / ".join(f"{v['total_ms']:.0f} ms {'GPU' if a == 'cuda' else 'CPU'}" for a, v in temps.items()),
            "temps_image_ms": temps.get("cuda", temps.get("cpu"))["total_ms"],
            "naif": m["test_reference_naive"]["erreur_moyenne_px"],
        })
    return lignes


def lignes_autres(n_test: int) -> list[dict]:
    if not AUTRES.exists():
        return []
    lignes = []
    for r in pd.read_csv(AUTRES, keep_default_na=False).to_dict("records"):
        if r.get("duree_totale_test_s") not in ("", None):
            ms = 1000 * float(r["duree_totale_test_s"]) / n_test
            r["temps_image"] = f"≈ {ms:.0f} ms ({r['machine']})*"
            r["temps_image_ms"] = ms
        else:
            r["temps_image"], r["temps_image_ms"] = "—", np.nan
        lignes.append(r)
    return lignes


# ---------------------------------------------------------------- tableau
COLONNES = [  # (clé, titre, format, meilleur)
    ("categorie", "Catégorie", None, None),
    ("methode", "Méthode", None, None),
    ("erreur_moyenne", "erreur\nmoyenne", "{:.1f} px", "min"),
    ("erreur_mediane", "erreur\nmédiane", "{:.1f} px", "min"),
    ("erreur_max", "erreur\nmax", "{:.0f} px", "min"),
    ("biais", "biais\nmoyen", "{:+.1f} px", "absmin"),
    ("a5", "à 5 px\nprès", "{:.1f} %", "max"),
    ("a10", "à 10 px\nprès", "{:.1f} %", "max"),
    ("a20", "à 20 px\nprès", "{:.1f} %", "max"),
    ("apprentissage", "apprentissage", None, None),
    ("duree", "durée\nd'entraînement", None, None),
    ("temps_image", "temps\npar image", None, None),
]
TEINTES = {"Lignes de pixels (sans apprentissage)": "#fdf1e6", "POD": "#edf5ec",
           "SAM 2.1-S (segmentation)": "#f1ecf8", CATEGORIE_ENC: "#e7f0fb", "Référence": "#ffffff"}
ENTETE, BORD, ENCRE, ENCRE_2 = "#e4e8f0", "#c3c2b7", "#0b0b0b", "#52514c"


def nombre(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


NOTES = ("En gras : meilleure valeur de la colonne (|biais| le plus petit) ; méthode en gras : plus petite erreur moyenne.   "
         "biais = prédiction − vérité (positif : ligne prédite trop bas).\n"
         "Durée d'entraînement : encodeurs = somme des époques (entraînement + validation) ; sans apprentissage = calcul "
         "complet sur le test.   Temps par image : une photo à la fois, médiane ; comprend ~19 ms de lecture du JPEG et "
         "découpe de la bande (CPU), le reste est le réseau.\n"
         "* temps par image déduit de la durée totale sur le test (durée / nombre de photos).   "
         "Valeurs des lignes de pixels, POD et SAM : encodeur/autres_methodes.csv.   Machines différentes.")


def dessiner(df: pd.DataFrame, n_test: int, sortie: Path, notes: str = NOTES):
    textes, gras = [], np.zeros((len(df), len(COLONNES)), dtype=bool)
    for j, (cle, _, fmt, meilleur) in enumerate(COLONNES):
        if meilleur:
            v = df[cle].map(nombre)
            v = v.abs() if meilleur == "absmin" else v
            cible = v.max() if meilleur == "max" else v.min()
            gras[:, j] = np.isclose(v, cible)
    for _, r in df.iterrows():
        ligne = []
        for cle, _, fmt, _ in COLONNES:
            v = r[cle]
            ligne.append("—" if v in ("", None) or (isinstance(v, float) and np.isnan(v))
                         else fmt.format(float(v)) if fmt else str(v))
        textes.append(ligne)
    # la catégorie n'est écrite qu'une fois par groupe, sur deux lignes si elle est longue
    for i in range(len(df) - 1, -1, -1):
        if i > 0 and df.categorie.iat[i] == df.categorie.iat[i - 1]:
            textes[i][0] = ""
        else:
            textes[i][0] = textwrap.fill(textes[i][0], 24)
    meilleure = df.erreur_moyenne.map(nombre).idxmin()

    largeurs = [0.13, 0.17, 0.065, 0.065, 0.06, 0.065, 0.06, 0.06, 0.06, 0.085, 0.12, 0.13]
    haut_lignes = 0.48 * (len(df) + 1)                     # pouces occupés par le tableau
    fig = plt.figure(figsize=(22, haut_lignes + 1.5))
    ax = fig.add_axes((0.005, 1.0 / (haut_lignes + 1.5), 0.99, haut_lignes / (haut_lignes + 1.5)))
    ax.axis("off")
    tab = ax.table(cellText=textes, colLabels=[c[1] for c in COLONNES], colWidths=largeurs,
                   cellLoc="center", bbox=(0, 0, 1, 1))
    tab.auto_set_font_size(False)
    tab.set_fontsize(10.5)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor(BORD)
        if i == 0:
            cell.set_facecolor(ENTETE)
            cell.set_text_props(weight="bold", color=ENCRE)
            continue
        r = df.iloc[i - 1]
        cell.set_facecolor(TEINTES.get(r.categorie, "#ffffff"))
        poids = "bold" if (j < 2 and (j == 0 or i - 1 == meilleure)) or (j >= 2 and gras[i - 1, j]) else "normal"
        cell.set_text_props(weight=poids, color=ENCRE_2 if j == 0 else ENCRE,
                            fontsize=9.5 if j == 0 else 10.5)
        if j == 1:
            cell.set_text_props(ha="left")
            cell.PAD = 0.04
    fig.text(0.005, 1 - 0.12 / (haut_lignes + 1.5), f"Hauteur d'eau sur les {n_test} photos test : toutes les méthodes",
             fontsize=15, color=ENCRE, va="top")
    fig.text(0.005, 0.1 / (haut_lignes + 1.5), notes, fontsize=9, color=ENCRE_2, va="bottom")
    fig.savefig(sortie, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="photos test pour mesurer le temps par image (défaut 100)")
    p.add_argument("--sans-cpu", action="store_true", help="ne pas mesurer le temps sur CPU")
    a = p.parse_args()

    enc = lignes_encodeurs(a.n, not a.sans_cpu)
    if not enc:
        raise SystemExit("Aucun encodeur entraîné : lancer d'abord encodeur/entrainer.py")
    n_test = int(json.loads((RESULTATS / "dinov2" / "metriques.json").read_text())["test"]["n"]) \
        if (RESULTATS / "dinov2" / "metriques.json").exists() else 883
    naif = {"categorie": "Référence", "methode": "Niveau moyen de train", "erreur_moyenne": enc[0]["naif"],
            "apprentissage": "—", "duree": "—", "temps_image": "—"}
    df = pd.DataFrame(lignes_autres(n_test) + enc + [naif]).drop(columns=["naif"], errors="ignore")
    for cle, *_ in COLONNES:
        if cle not in df:
            df[cle] = ""
    df = df.astype(object).fillna("")

    df.to_csv(RESULTATS / "tableau_resultats.csv", index=False,
              columns=[c[0] for c in COLONNES] + ["temps_image_ms"])
    dessiner(df.reset_index(drop=True), n_test, RESULTATS / "tableau_resultats.png")

    print("\nEncodeurs :")
    for r in enc:
        print(f"  {r['methode']:24s} erreur moyenne {r['erreur_moyenne']:.1f} px | entraînement {r['duree']} | "
              f"temps par image {r['temps_image']}")
    print(f"\n-> {(RESULTATS / 'tableau_resultats.png').relative_to(ICI.parent)}"
          f"\n-> {(RESULTATS / 'tableau_resultats.csv').relative_to(ICI.parent)}")


if __name__ == "__main__":
    main()
