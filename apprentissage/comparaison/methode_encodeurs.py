"""Encodeurs pré-entraînés (dossier encodeur/) : réentraînement puis temps par image.

    .venv/bin/python comparaison/methode_encodeurs.py                        # resnet50, convnext, dinov2
    .venv/bin/python comparaison/methode_encodeurs.py --encodeurs dinov2
    .venv/bin/python comparaison/methode_encodeurs.py --sans-reentrainer     # garde les modèles déjà entraînés

Durée d'entraînement = durée totale de encodeur/entrainer.py (chargement, époques avec validation, test).
Temps par image = lecture du JPEG + bande + réseau, une photo à la fois, sur GPU (encodeur/tableau_resultats.py).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import pandas as pd
import torch

from commun import RESULTATS, ROOT, donnees, enregistrer, nom_gpu

from tableau_resultats import ENCODEURS, temps_par_image   # noqa: E402  (encodeur/)

CATEGORIE = "Encodeurs pré-entraînés"
DOSSIER = ROOT / "apprentissage" / "encodeurs" / "resultats"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encodeurs", nargs="+", default=list(ENCODEURS), choices=list(ENCODEURS))
    ap.add_argument("--sans-reentrainer", action="store_true", help="ne pas relancer l'entraînement")
    ap.add_argument("--n-temps", type=int, default=100, help="photos test pour le temps par image")
    a = ap.parse_args()

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    photos = donnees().query("split == 'test'").sample(frac=1, random_state=0).head(a.n_temps)
    for cle in a.encodeurs:
        duree = None
        if not a.sans_reentrainer:
            print(f"Entraînement de {cle} (journal : encodeur/resultats/journal_{cle}.txt)...", flush=True)
            t0 = time.perf_counter()
            with open(DOSSIER.parent / "resultats" / f"journal_{cle}.txt", "w") as journal:
                subprocess.run([sys.executable, "-u", str(ROOT / "apprentissage" / "encodeurs" / "entrainer.py"), "--encodeur", cle],
                               stdout=journal, stderr=subprocess.STDOUT, check=True)
            duree = time.perf_counter() - t0
        else:
            duree = pd.read_csv(DOSSIER / cle / "historique.csv").duree_s.sum()
        m = json.loads((DOSSIER / cle / "metriques.json").read_text())
        temps = temps_par_image(cle, photos, appareil)
        p = pd.read_csv(DOSSIER / cle / "predictions_test.csv")
        enregistrer(f"encodeur_{cle}", CATEGORIE, ENCODEURS[cle], p, apprentissage=f"{m['meilleure_epoque']} époques",
                    duree_entrainement_s=duree, machine_entrainement=nom_gpu(), temps_image_ms=temps["total_ms"],
                    machine_image=nom_gpu() if appareil.type == "cuda" else "CPU",
                    temps_preparation_ms=temps["preparation_ms"], temps_reseau_ms=temps["reseau_ms"])


if __name__ == "__main__":
    main()
