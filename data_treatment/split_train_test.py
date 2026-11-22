"""Découpe toutes les photos du catalogue en deux ensembles, train et test, par semaines entières.

    .venv/bin/python data_treatment/split_train_test.py                   # 4 semaines train, 1 semaine test
    .venv/bin/python data_treatment/split_train_test.py --train 2 --test 1

Semaine = semaine du calendrier (lundi -> dimanche), numérotée à partir de la première semaine de la
saison. Les semaines se suivent par cycles : les --train premières
semaines du cycle vont en train, les --test suivantes en test. Une semaine entière va dans le même
ensemble : les photos d'une même semaine (même niveau d'eau, même lumière) ne se retrouvent jamais
des deux côtés.

Le découpage porte sur toutes les photos (pas seulement les annotées) : il ne change pas quand on
annote, et une nouvelle semaine de photos ne change pas les anciennes. Le catalogue n'est pas modifié.

Sorties :
    data_treatment/split_train_test.csv   (fichier, semaine, split)
    data/dataset/training/  et  data/dataset/test/   les images de chaque ensemble
        Liens physiques vers data/images/ : pas d'espace disque en plus, et data/images/ n'est pas
        modifié. À chaque lancement, ces deux dossiers sont vidés puis refaits (une image ne peut pas
        rester dans les deux). Ne pas retoucher ces images : un lien physique, c'est le même fichier
        que dans data/images/. --sans-dossiers pour ne produire que le CSV.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ICI = Path(__file__).resolve().parent
CATALOGUE = ICI / "catalogue_images.csv"
ANNOTATIONS = ICI / "annotations"
SORTIE = ICI / "split_train_test.csv"
IMAGES = ICI.parent / "data" / "images"
DATASET = ICI.parent / "data" / "dataset"
DOSSIERS = {"train": "training", "test": "test"}


def decouper(fichiers: pd.Series, n_train: int, n_test: int) -> pd.DataFrame:
    """Semaine et ensemble (train / test) de chaque photo, d'après la date dans son nom."""
    date = pd.to_datetime(fichiers.str[:10])
    lundi = date - pd.to_timedelta(date.dt.day_of_week, unit="D")
    semaine = ((lundi - lundi.min()).dt.days // 7).astype(int)
    split = np.where(semaine % (n_train + n_test) < n_train, "train", "test")
    return pd.DataFrame({"fichier": fichiers, "semaine": semaine, "split": split})


def creer_dossiers(split: pd.DataFrame) -> dict[str, int]:
    """data/dataset/training/ et test/ : liens physiques vers data/images/, refaits à chaque fois."""
    for nom in DOSSIERS.values():
        dossier = DATASET / nom
        dossier.mkdir(parents=True, exist_ok=True)
        for f in dossier.glob("*.jpg"):      # on retire les anciens liens (les originaux restent dans data/images/)
            f.unlink()
    for r in split.itertuples():
        src, dest = IMAGES / r.fichier, DATASET / DOSSIERS[r.split] / r.fichier
        try:
            os.link(src, dest)
        except OSError:                      # autre disque, système de fichiers sans liens...
            shutil.copy2(src, dest)
    return {nom: len(list((DATASET / nom).glob("*.jpg"))) for nom in DOSSIERS.values()}


def annotees() -> pd.DataFrame:
    """Photos dont le niveau est annoté (tous les fichiers annotations/niveau_*.csv)."""
    parts = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in sorted(ANNOTATIONS.glob("niveau_*.csv"))]
    if not parts:
        return pd.DataFrame(columns=["fichier", "niveau_px"])
    a = pd.concat(parts, ignore_index=True)
    return a[a.niveau_px != ""].drop_duplicates("fichier")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=int, default=4, help="semaines de train par cycle (défaut 4)")
    p.add_argument("--test", type=int, default=1, help="semaines de test par cycle (défaut 1)")
    p.add_argument("--sortie", type=Path, default=SORTIE)
    p.add_argument("--sans-dossiers", action="store_true", help="ne pas créer data/dataset/training et test")
    a = p.parse_args()

    cat = pd.read_csv(CATALOGUE, dtype=str, keep_default_na=False)
    split = decouper(cat.fichier, a.train, a.test)
    split.to_csv(a.sortie, index=False)
    if not a.sans_dossiers:
        n = creer_dossiers(split)
        print(f"Images : {n} dans {DATASET.relative_to(ICI.parent)}/")

    # Bilan
    cat = cat.merge(split, on="fichier")
    cat["annotee"] = cat.fichier.isin(annotees().fichier)
    print(f"Cycle : {a.train} semaine(s) train, {a.test} semaine(s) test  ->  {a.sortie.name}\n")
    bilan = pd.DataFrame({
        "photos": cat.groupby("split").size(),
        "exploitables": cat[cat.exploitable != "non"].groupby("split").size(),
        "annotées": cat[cat.annotee].groupby("split").size(),
        "semaines avec photos": cat.groupby("split").semaine.nunique(),
    }).fillna(0).astype(int).loc[["train", "test"]]
    pct = (100 * bilan.loc["test"] / bilan.sum()).round(1).astype(str) + " %"
    print(pd.concat([bilan.astype(str), pct.to_frame("% test").T]).to_string(), "\n")
    sem = cat.groupby("semaine").agg(debut=("jour", "min"), fin=("jour", "max"), photos=("fichier", "size"),
                                     annotees=("annotee", "sum"), split=("split", "first"))
    print("Semaines de test :")
    print(sem[sem.split == "test"].drop(columns="split").to_string())


if __name__ == "__main__":
    main()
