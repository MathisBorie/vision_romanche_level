"""Classe chaque photo de catalogue_images.csv à partir des mesures de standardisation.py.

Étape 3 de la chaîne : restructure.py -> standardisation.py -> classification.py -> define_water_lvl.py
    .venv/bin/python data_treatment/classification.py

Ne relit pas les images (instantané) : pour essayer d'autres seuils, les modifier ci-dessous et relancer.

Colonnes écrites :
    periode         jour / crepuscule / nuit (d'après la luminosité ; l'heure ne suffit pas : saisons)
    visibilite      nette / brumeuse / voilee (brouillard ou objectif voilé par le soleil), vide la nuit
    reflets         1 si pct_sature >= 2 %
    exploitable     oui / douteux / non  (non = nuit ou taille inconnue ; voilée, brumeuse, reflets,
                    crépuscule ou alignement non fiable = douteux, à trier pendant l'annotation du niveau)

Colonnes manuelles (créées vides si besoin, jamais écrasées) :
    niveau_px       y de la frontière eau / berge d'en face, en px de l'image d'origine
    classe_niveau   bas / moyen / haut / crue
    verifie         1 si un humain a vérifié la ligne : periode, visibilite, reflets et exploitable
                    de cette ligne ne sont alors plus recalculés (on peut les corriger à la main)
    commentaire
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # racine du projet (data/)
CSV = Path(__file__).resolve().parent / "catalogue_images.csv"   # à côté des scripts

# Seuils, réglés en regardant des planches d'images (mesures calculées à 1/4 de résolution)
LUM_NUIT, LUM_JOUR = 20, 80           # luminosite < 20 : nuit ; < 80 : crépuscule
NET_VOILEE, NET_BRUMEUSE = 0.35, 1.5  # nettete < 0.35 : voilée ; < 1.5 : brumeuse
SATURE_REFLETS = 2.0                  # pct_sature >= 2 % : reflets
CONFIANCE_MIN = 0.2                   # confiance_alignement en dessous : décalage non fiable

CLASSES = ["periode", "visibilite", "reflets", "exploitable"]
MANUELLES = ["niveau_px", "classe_niveau", "verifie", "commentaire"]
MESURES = ["luminosite", "nettete", "pct_sature", "confiance_alignement"]


def classer(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les classes à partir des mesures (df en texte, comme lu dans le CSV)."""
    m = {c: pd.to_numeric(df[c], errors="coerce") for c in MESURES}
    out = pd.DataFrame(index=df.index)

    out["periode"] = np.select([m["luminosite"] < LUM_NUIT, m["luminosite"] < LUM_JOUR],
                               ["nuit", "crepuscule"], "jour")
    nuit = out.periode == "nuit"
    out["visibilite"] = np.select([nuit, m["nettete"] < NET_VOILEE, m["nettete"] < NET_BRUMEUSE],
                                  ["", "voilee", "brumeuse"], "nette")
    out["reflets"] = (m["pct_sature"] >= SATURE_REFLETS).astype(int).astype(str)

    # les images voilées restent "douteux" : elles seront triées à la main pendant l'annotation du niveau
    non = (df.format == "autre") | nuit
    douteux = ((out.periode == "crepuscule") | out.visibilite.isin(["brumeuse", "voilee"])
               | (out.reflets == "1") | ~(m["confiance_alignement"] >= CONFIANCE_MIN))
    out["exploitable"] = np.select([non, douteux], ["non", "douteux"], "oui")
    return out


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"{CSV.name} absent : lancer standardisation.py d'abord")
    df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    manquantes = [c for c in MESURES + ["format"] if c not in df]
    if manquantes:
        raise ValueError(f"Colonnes de mesure absentes {manquantes} : lancer standardisation.py d'abord")

    for c in CLASSES + MANUELLES:
        if c not in df:
            df[c] = ""
    a_classer = df.verifie != "1"             # les lignes vérifiées à la main gardent leurs classes
    df.loc[a_classer, CLASSES] = classer(df)[a_classer]

    autres = [c for c in df.columns if c not in CLASSES + MANUELLES]
    df = df[autres + CLASSES + MANUELLES]
    df.to_csv(CSV, index=False)

    print(f"{len(df)} images classées -> {CSV.name} ({(~a_classer).sum()} lignes vérifiées à la main conservées)")
    for c in ["periode", "visibilite", "exploitable"]:
        print(f"  {c:12s}", df[c].replace("", "-").value_counts().to_dict())
    print(f"  reflets      {(df.reflets == '1').sum()}")
    print(f"  niveaux annotés : {(df.niveau_px != '').sum()}")


if __name__ == "__main__":
    main()
