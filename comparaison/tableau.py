"""Tableau comparatif à partir de comparaison/resultats/*.json (toutes les méthodes, mesurées sur ce PC).

    .venv/bin/python comparaison/tableau.py

Sorties : comparaison/resultats/tableau_comparaison.png et .csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from commun import RESULTATS, donnees, nom_cpu, nom_gpu

from tableau_resultats import COLONNES, dessiner   # noqa: E402  (encodeur/)

ORDRE = ["lignes_rgb", "lignes_canny", "lignes_texture", "pod_lineaire_750", "pod_reseau_300", "pod_reseau_750",
         "sam_zero_shot", "sam_affine", "encodeur_resnet50", "encodeur_convnext", "encodeur_dinov2"]


def duree(s) -> str:
    if s is None:
        return "—"
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


def temps(ms: float, machine: str) -> str:
    return f"{ms:.1f} ms ({machine})" if ms < 10 else f"{ms:.0f} ms ({machine})"


def main():
    res = {f.stem: json.loads(f.read_text()) for f in RESULTATS.glob("*.json")}
    manquent = [c for c in ORDRE if c not in res]
    if manquent:
        print(f"Pas encore de résultat pour : {', '.join(manquent)} (lignes absentes du tableau)")
    lignes = []
    for cle in [c for c in ORDRE if c in res] + sorted(set(res) - set(ORDRE)):
        r = res[cle]
        t = r["test"]
        machine_ent = r.get("machine_entrainement") or ""
        lignes.append({
            "categorie": r["categorie"], "methode": r["methode"], **{k: t[k] for k in
            ("erreur_moyenne", "erreur_mediane", "erreur_max", "biais", "a5", "a10", "a20")},
            "apprentissage": r["apprentissage"],
            "duree": "—" if r["duree_entrainement_s"] is None else f"{duree(r['duree_entrainement_s'])} ({machine_ent})",
            "temps_image": temps(r["temps_image_ms"], r["machine_image"]), "temps_image_ms": r["temps_image_ms"],
            "n": t["n"], "sans_prediction": r.get("sans_prediction", 0),
        })
    df = donnees()
    test, train = df[df.split == "test"], df[df.split == "train"]
    naif = float(np.abs(test.niveau_px - train.niveau_px.mean()).mean())
    lignes.append({"categorie": "Référence", "methode": "Niveau moyen de train", "erreur_moyenne": round(naif, 1),
                   "apprentissage": "—", "duree": "—", "temps_image": "—"})
    tab = pd.DataFrame(lignes)
    for cle, *_ in COLONNES:
        if cle not in tab:
            tab[cle] = ""
    tab = tab.astype(object).fillna("").reset_index(drop=True)

    tab.to_csv(RESULTATS / "tableau_comparaison.csv", index=False,
               columns=[c[0] for c in COLONNES] + ["temps_image_ms", "n", "sans_prediction"])
    notes = (f"Tout mesuré sur le même PC : CPU {nom_cpu()}, GPU {nom_gpu()} ; mêmes {len(test)} photos test (semaines jamais "
             "vues), même vérité (annotations), même train / val.   En gras : meilleure valeur de la colonne (|biais| le "
             "plus petit) ; méthode en gras : plus petite erreur moyenne.\n"
             "biais = prédiction − vérité (positif : ligne prédite trop bas).   Durée d'entraînement : du lancement au modèle "
             "prêt (POD : lecture des photos + POD + ajustement ; SAM affiné et encodeurs : époques avec validation + test).\n"
             "Temps par image : une photo à la fois (lot de 1), lecture du JPEG comprise, médiane ; méthodes CPU limitées à "
             "1 cœur, méthodes GPU sur la carte graphique.   POD : K fixé d'avance (pas choisi sur le test).")
    sortie = RESULTATS / "tableau_comparaison.png"
    dessiner(tab, len(test), sortie, notes=notes)
    print(f"-> {sortie.relative_to(RESULTATS.parent.parent)}\n-> {(RESULTATS / 'tableau_comparaison.csv').relative_to(RESULTATS.parent.parent)}")


if __name__ == "__main__":
    main()
