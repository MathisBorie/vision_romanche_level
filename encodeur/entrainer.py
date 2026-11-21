"""Entraîne un détecteur du niveau d'eau (frontière eau / berge) sur un encodeur pré-entraîné.

    .venv/bin/python encodeur/entrainer.py --encodeur dinov2            # DINOv2 ViT-S gelé (recommandé)
    .venv/bin/python encodeur/entrainer.py --encodeur convnext          # ConvNeXt-Tiny affiné
    .venv/bin/python encodeur/entrainer.py --encodeur resnet50          # ResNet-50 affiné
    .venv/bin/python encodeur/entrainer.py --encodeur dinov2 --limite 64 --epoques 1   # essai rapide

- Entraînement sur train, choix de la meilleure époque sur val (semaines prises dans train),
  évaluation finale sur test (semaines jamais vues).
- Erreur mesurée en px de l'image d'origine, sur niveau_px.
- Résultats dans encodeur/resultats/<encodeur>/ :
    metriques.json, historique.csv, predictions_test.csv (versionnables), modele.pt (ignoré par git).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from donnees import Y0, Y1, BandeRiviere, charger_annotations, vers_px
from modele import ENCODEURS, DetecteurNiveau

ICI = Path(__file__).resolve().parent


def metriques(err: np.ndarray) -> dict:
    a = np.abs(err)
    return {"n": int(len(a)), "erreur_moyenne_px": round(float(a.mean()), 2),
            "erreur_mediane_px": round(float(np.median(a)), 2), "erreur_max_px": round(float(a.max()), 1),
            "a_5px_pct": round(100 * float((a <= 5).mean()), 1), "a_10px_pct": round(100 * float((a <= 10).mean()), 1),
            "a_20px_pct": round(100 * float((a <= 20).mean()), 1)}


@torch.no_grad()
def predire(modele, chargeur, appareil) -> pd.DataFrame:
    modele.eval()
    lignes = []
    for x, y_rel, haut, fichiers in chargeur:
        with torch.autocast(appareil.type, dtype=torch.bfloat16, enabled=appareil.type == "cuda"):
            pred, _ = modele(x.to(appareil, non_blocking=True))
        lignes.append(pd.DataFrame({"fichier": fichiers, "niveau_px": vers_px(y_rel, haut),
                                    "prediction_px": vers_px(pred.float().cpu(), haut)}))
    df = pd.concat(lignes, ignore_index=True)
    df["erreur_px"] = df.prediction_px - df.niveau_px
    return df.round(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encodeur", choices=list(ENCODEURS), default="dinov2")
    p.add_argument("--epoques", type=int, default=20)
    p.add_argument("--lot", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="taux d'apprentissage de la tête")
    p.add_argument("--lr-encodeur", type=float, default=5e-5, help="taux pour un encodeur affiné")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limite", type=int, default=0, help="n'utiliser que N photos par ensemble (essai rapide)")
    p.add_argument("--graine", type=int, default=0)
    a = p.parse_args()

    random.seed(a.graine); np.random.seed(a.graine); torch.manual_seed(a.graine)
    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sortie = ICI / "resultats" / a.encodeur
    sortie.mkdir(parents=True, exist_ok=True)

    df = charger_annotations()
    ens = {s: df[df.split == s] for s in ("train", "val", "test")}
    if a.limite:
        ens = {s: d.sample(min(a.limite, len(d)), random_state=0) for s, d in ens.items()}
    print(f"{a.encodeur} ({ENCODEURS[a.encodeur]['nom']}) sur {appareil} | "
          + ", ".join(f"{s} {len(d)} photos / {d.semaine.nunique()} semaines" for s, d in ens.items()))

    charge = lambda d, aug, melange: DataLoader(BandeRiviere(d, augmenter=aug), batch_size=a.lot, shuffle=melange,
                                                num_workers=a.workers, pin_memory=True, drop_last=melange,
                                                persistent_workers=a.workers > 0)
    c_train, c_val, c_test = charge(ens["train"], True, True), charge(ens["val"], False, False), charge(ens["test"], False, False)

    modele = DetecteurNiveau(a.encodeur).to(appareil)
    groupes = [{"params": modele.tete.parameters(), "lr": a.lr}]
    if not modele.encodeur.gele:
        groupes.append({"params": modele.encodeur.parameters(), "lr": a.lr_encodeur})
    opt = torch.optim.AdamW(groupes, weight_decay=1e-4)
    etapes = a.epoques * len(c_train)
    planif = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groupes], total_steps=etapes,
                                                 pct_start=0.1)

    # référence naïve : toujours prédire le niveau moyen de train
    naif = metriques(ens["test"].niveau_px.to_numpy() - ens["train"].niveau_px.mean())
    print(f"référence naïve (niveau moyen de train) sur test : erreur moyenne {naif['erreur_moyenne_px']} px")

    historique, meilleure = [], np.inf
    for ep in range(1, a.epoques + 1):
        modele.train()
        t0, pertes = time.time(), []
        for x, y_rel, _, _ in c_train:
            x, y_rel = x.to(appareil, non_blocking=True), y_rel.to(appareil)
            with torch.autocast(appareil.type, dtype=torch.bfloat16, enabled=appareil.type == "cuda"):
                pred, _ = modele(x)
            perte = F.smooth_l1_loss(pred.float() * (Y1 - Y0), y_rel * (Y1 - Y0), beta=2.0)   # en px
            opt.zero_grad(set_to_none=True)
            perte.backward()
            opt.step(); planif.step()
            pertes.append(perte.item())
        val = metriques(predire(modele, c_val, appareil).erreur_px.to_numpy())
        historique.append({"epoque": ep, "perte_train_px": round(float(np.mean(pertes)), 2),
                           "val_erreur_moyenne_px": val["erreur_moyenne_px"],
                           "val_erreur_mediane_px": val["erreur_mediane_px"], "duree_s": round(time.time() - t0)})
        print(f"époque {ep:2d}/{a.epoques}  perte train {np.mean(pertes):6.2f} px | val : moyenne "
              f"{val['erreur_moyenne_px']:6.2f} px, médiane {val['erreur_mediane_px']:5.2f} px  ({time.time() - t0:.0f} s)")
        if val["erreur_moyenne_px"] < meilleure:
            meilleure = val["erreur_moyenne_px"]
            torch.save({"modele": modele.state_dict(), "encodeur": a.encodeur, "epoque": ep}, sortie / "modele.pt")

    # évaluation finale sur test avec la meilleure époque
    etat = torch.load(sortie / "modele.pt", map_location=appareil)
    modele.load_state_dict(etat["modele"])
    pred_test = predire(modele, c_test, appareil)
    res = {"encodeur": a.encodeur, "modele_timm": ENCODEURS[a.encodeur]["nom"], "meilleure_epoque": etat["epoque"],
           "val": metriques(predire(modele, c_val, appareil).erreur_px.to_numpy()),
           "test": metriques(pred_test.erreur_px.to_numpy()), "test_reference_naive": naif,
           "parametres": vars(a) | {"photos": {s: len(d) for s, d in ens.items()}}}
    pred_test.to_csv(sortie / "predictions_test.csv", index=False)
    pd.DataFrame(historique).to_csv(sortie / "historique.csv", index=False)
    (sortie / "metriques.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    t = res["test"]
    print(f"\nTEST ({t['n']} photos, époque {etat['epoque']}) : erreur moyenne {t['erreur_moyenne_px']} px, "
          f"médiane {t['erreur_mediane_px']} px, max {t['erreur_max_px']} px | "
          f"{t['a_5px_pct']} % à 5 px, {t['a_10px_pct']} % à 10 px, {t['a_20px_pct']} % à 20 px")
    print(f"Résultats dans {sortie.relative_to(ICI.parent)}/")


if __name__ == "__main__":
    main()
