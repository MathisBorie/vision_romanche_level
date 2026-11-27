"""Outils communs à toutes les méthodes : mêmes données, mêmes métriques, même format de résultats.

- Données : encodeur/donnees.py (annotations de data_treatment/annotations/, découpage
  data_treatment/split_train_test.csv). test = les mêmes photos annotées pour toutes les méthodes.
  val = une semaine de train sur quatre (pour les méthodes qui choisissent une époque).
- Chaque méthode enregistre comparaison/resultats/<cle>.json et <cle>_predictions.csv.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ICI = Path(__file__).resolve().parent
ROOT = ICI.parent
RESULTATS = ICI / "resultats"
sys.path.insert(0, str(ROOT / "encodeur"))
sys.path.insert(0, str(ROOT))

from donnees import charger_annotations  # noqa: E402  (encodeur/donnees.py)


def donnees() -> pd.DataFrame:
    """fichier, niveau_px, annotateur, semaine, split (train / val / test), decalage_y."""
    return charger_annotations()


def nom_cpu() -> str:
    try:
        for ligne in Path("/proc/cpuinfo").read_text().splitlines():
            if ligne.startswith("model name"):
                return ligne.split(":", 1)[1].replace("12th Gen Intel(R) Core(TM)", "").strip()
    except OSError:
        pass
    return platform.processor() or "CPU"


def nom_gpu() -> str:
    import torch
    return torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "") if torch.cuda.is_available() else nom_cpu()


def metriques(err: np.ndarray) -> dict:
    err = np.asarray(err, dtype=float)
    a = np.abs(err)
    return {"n": int(len(a)), "erreur_moyenne": round(float(a.mean()), 2),
            "erreur_mediane": round(float(np.median(a)), 2), "erreur_max": round(float(a.max()), 1),
            "biais": round(float(err.mean()), 2), "a5": round(100 * float((a <= 5).mean()), 1),
            "a10": round(100 * float((a <= 10).mean()), 1), "a20": round(100 * float((a <= 20).mean()), 1)}


class Chrono:
    """Mesure le temps de chaque appel : with chrono: ... ; chrono.mediane_ms()."""

    def __init__(self):
        self.durees = []

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.durees.append(time.perf_counter() - self.t0)

    def mediane_ms(self) -> float:
        return round(1000 * float(np.median(self.durees)), 1)

    def total_s(self) -> float:
        return round(float(np.sum(self.durees)), 1)


def enregistrer(cle: str, categorie: str, methode: str, predictions: pd.DataFrame, apprentissage: str,
                duree_entrainement_s: float | None, temps_image_ms: float, machine_image: str,
                machine_entrainement: str | None = None, **autres) -> dict:
    """predictions : fichier, niveau_px, prediction_px. Écrit <cle>.json et <cle>_predictions.csv."""
    RESULTATS.mkdir(parents=True, exist_ok=True)
    p = predictions[["fichier", "niveau_px", "prediction_px"]].copy()
    p["erreur_px"] = (p.prediction_px - p.niveau_px).round(1)
    manquantes = int(p.prediction_px.isna().sum())
    if manquantes:                      # pas de prédiction (ex. SAM ne trouve pas d'eau) : on la compte au pire
        print(f"  {cle} : {manquantes} photos sans prédiction (exclues des métriques)")
    p.to_csv(RESULTATS / f"{cle}_predictions.csv", index=False)
    res = {"cle": cle, "categorie": categorie, "methode": methode, "test": metriques(p.erreur_px.dropna()),
           "sans_prediction": manquantes, "apprentissage": apprentissage,
           "duree_entrainement_s": None if duree_entrainement_s is None else round(duree_entrainement_s, 1),
           "machine_entrainement": machine_entrainement, "temps_image_ms": temps_image_ms,
           "machine_image": machine_image, **autres}
    (RESULTATS / f"{cle}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    t = res["test"]
    print(f"  {methode:28s} erreur moyenne {t['erreur_moyenne']:6.1f} px  médiane {t['erreur_mediane']:5.1f}  "
          f"max {t['erreur_max']:5.0f}  biais {t['biais']:+5.1f} | entraînement "
          f"{'—' if duree_entrainement_s is None else f'{duree_entrainement_s / 60:.1f} min'} | "
          f"{temps_image_ms:.1f} ms/image ({machine_image})", flush=True)
    return res
