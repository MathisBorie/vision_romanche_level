"""Affinage de SAM 2 sur les niveaux annotés : seul le décodeur de masques apprend.

On n'a pas de masques d'eau, seulement un niveau y par photo. La perte porte donc directement sur la
ligne déduite du masque, de façon dérivable :
    p(r, c) = sigmoid(NETTETE x logit) du masque basse résolution (256 x 256, une rangée = 6,85 px)
              (les logits de SAM sont faibles, ~ -2 / +2 : sans NETTETE la ligne remonte de ~400 px)
    S(r, c) = prod_{r' <= r} (1 - p(r', c))   probabilité qu'il n'y ait pas encore d'eau jusqu'à la rangée r
    y(c)    = haut de la bande + somme_r S(r, c) x hauteur d'une rangée   (position attendue de la 1re eau)
    y       = moyenne de y(c) sur les colonnes
Si le masque est net, y(c) est exactement le bord supérieur de l'eau (comme sam_bordure.py) ; les trous
dans l'eau sous la frontière (rochers, écume) ne comptent pas.
Perte : smooth L1 en px (beta 2), comme encodeur/entrainer.py.

- Encodeur d'image (Hiera) gelé, prompt fixe identique à sam_bordure.py (points eau / berge + boîte).
- Données : evaluation/split_encodeur.csv (train / val / test du dossier encodeur/, identiques).
- Meilleure époque choisie sur val ; test seulement pour l'évaluation finale.

    .venv/bin/python sam_affiner.py                      # entraînement complet
    .venv/bin/python sam_affiner.py --limite 32 --epoques 1   # essai rapide

Sorties (evaluation/sam_affine/) : metriques.json, historique.csv, predictions_test.csv,
decodeur.pt (poids du décodeur, ignoré par git).
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sam_bordure import BOITE, MODELE, X_POINTS, Y_BERGE, Y_EAU, choisir_device, metriques

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir())
ICI = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "images"
SPLIT = ICI / "evaluation" / "split_encodeur.csv"
SORTIE = ICI / "evaluation" / "sam_affine"

W0, H0 = 1919, 1753                 # image d'origine
COTE = 1024                         # entrée de SAM (redimensionnement carré, comme Sam2Processor)
MASQUE = 256                        # côté du masque basse résolution
H_RANG = H0 / MASQUE                # hauteur d'une rangée du masque, en px d'origine
Y0, Y1 = 300, 1350                  # bande où l'on cherche la frontière (comme encodeur/)
R0, R1 = int(Y0 // H_RANG), int(np.ceil(Y1 / H_RANG))
NETTETE = 16.0                      # facteur appliqué aux logits avant la sigmoïde
MOYENNE, ECART_TYPE = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def prompts(n: int, device: str):
    """Prompt fixe, en coordonnées de l'entrée 1024 x 1024."""
    sx, sy = COTE / W0, COTE / H0
    pts = [[x * sx, Y_EAU * sy] for x in X_POINTS] + [[x * sx, Y_BERGE * sy] for x in X_POINTS]
    lab = [1] * len(X_POINTS) + [0] * len(X_POINTS)
    boite = [BOITE[0] * sx, BOITE[1] * sy, BOITE[2] * sx, BOITE[3] * sy]
    t = lambda v, dt: torch.tensor(v, dtype=dt, device=device)
    return (t(pts, torch.float32).expand(n, 1, -1, -1), t(lab, torch.long).expand(n, 1, -1),
            t(boite, torch.float32).expand(n, 1, -1))


class Photos(Dataset):
    def __init__(self, df: pd.DataFrame, augmenter: bool = False):
        self.df = df.reset_index(drop=True)
        self.augmenter = augmenter
        self.couleur = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.03)
        self.normaliser = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MOYENNE, ECART_TYPE)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(IMAGES / r.fichier)
        im.draft("RGB", (W0 // 2, H0 // 2))                 # décodage JPEG à 1/2 : plus rapide
        im = im.convert("RGB").resize((COTE, COTE), Image.BILINEAR)
        if self.augmenter:
            im = self.couleur(im)
            if random.random() < 0.5:                         # prompt quasi symétrique, ligne inchangée
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return self.normaliser(im), torch.tensor(float(r.niveau_px)), r.fichier


def niveau_doux(logits: torch.Tensor) -> torch.Tensor:
    """(B, 256, 256) logits du masque -> y en px d'origine, dérivable."""
    log_non_eau = F.logsigmoid(-NETTETE * logits[:, R0:R1, :].float())          # log(1 - p)
    survie = torch.exp(torch.cumsum(log_non_eau, dim=1))
    return R0 * H_RANG + survie.sum(dim=1).mean(dim=1) * H_RANG


def predire(model, pixels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        emb = model.get_image_embeddings(pixels)
    pts, lab, boite = prompts(len(pixels), pixels.device)
    out = model(image_embeddings=emb, input_points=pts, input_labels=lab, input_boxes=boite,
                multimask_output=False)
    return niveau_doux(out.pred_masks[:, 0, 0])


def evaluer(model, charge) -> pd.DataFrame:
    model.eval()
    lignes = []
    with torch.no_grad():
        for pixels, y, noms in charge:
            pred = predire(model, pixels.to(model.device)).cpu()
            lignes += [{"fichier": f, "niveau_px": float(v), "prediction_px": round(float(p), 1),
                        "erreur_px": round(float(p - v), 1)} for f, v, p in zip(noms, y, pred)]
    return pd.DataFrame(lignes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoques", type=int, default=8)
    ap.add_argument("--lot", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limite", type=int, default=0, help="N photos par ensemble (essai rapide)")
    ap.add_argument("--graine", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.graine); np.random.seed(a.graine); torch.manual_seed(a.graine)

    df = pd.read_csv(SPLIT)
    parts = {s: df[df.split == s] for s in ("train", "val", "test")}
    if a.limite:
        parts = {s: d.sample(min(a.limite, len(d)), random_state=0) for s, d in parts.items()}
    print({s: len(d) for s, d in parts.items()})
    charge = lambda s, aug, mel: DataLoader(Photos(parts[s], aug), batch_size=a.lot, shuffle=mel,
                                            num_workers=a.workers, persistent_workers=a.workers > 0)
    ch_train, ch_val, ch_test = charge("train", True, True), charge("val", False, False), charge("test", False, False)

    from transformers import Sam2Model
    device = choisir_device()
    model = Sam2Model.from_pretrained(MODELE).to(device)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("mask_decoder.")
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{MODELE} sur {device} ; décodeur entraîné : {sum(p.numel() for p in params) / 1e6:.1f} M paramètres")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    planif = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epoques * len(ch_train),
                                                 pct_start=0.1)
    SORTIE.mkdir(parents=True, exist_ok=True)

    m0 = metriques(evaluer(model, ch_val).erreur_px)       # époque 0 : SAM tel quel, ligne douce
    print(f"époque 0 (sans entraînement)  val {m0['erreur_moyenne_px']:.2f} px  biais {m0['biais_moyen_px']:+.2f}")
    historique, meilleure, t0 = [{"epoque": 0, "perte": np.nan, **{f"val_{k}": v for k, v in m0.items()}}], (m0["erreur_moyenne_px"], 0), time.time()
    torch.save({k: v.cpu() for k, v in model.mask_decoder.state_dict().items()}, SORTIE / "decodeur.pt")

    for ep in range(1, a.epoques + 1):
        model.train()
        pertes = []
        for i, (pixels, y, _) in enumerate(ch_train, 1):
            pred = predire(model, pixels.to(device))
            perte = F.smooth_l1_loss(pred, y.to(device).float(), beta=2.0)
            opt.zero_grad(); perte.backward(); opt.step(); planif.step()
            pertes.append(perte.item())
            if i % 100 == 0:
                print(f"  époque {ep}  lot {i}/{len(ch_train)}  perte {np.mean(pertes[-100:]):.2f}", flush=True)
        m = metriques(evaluer(model, ch_val).erreur_px)
        historique.append({"epoque": ep, "perte": np.mean(pertes), **{f"val_{k}": v for k, v in m.items()}})
        print(f"époque {ep}  perte {np.mean(pertes):.2f}  val {m['erreur_moyenne_px']:.2f} px  "
              f"médiane {m['erreur_mediane_px']:.2f}  biais {m['biais_moyen_px']:+.2f}  "
              f"({(time.time() - t0) / 60:.0f} min)", flush=True)
        if m["erreur_moyenne_px"] < meilleure[0]:
            meilleure = (m["erreur_moyenne_px"], ep)
            torch.save({k: v.cpu() for k, v in model.mask_decoder.state_dict().items()}, SORTIE / "decodeur.pt")
        pd.DataFrame(historique).to_csv(SORTIE / "historique.csv", index=False)

    model.mask_decoder.load_state_dict(torch.load(SORTIE / "decodeur.pt"))
    test = evaluer(model, ch_test)
    test.to_csv(SORTIE / "predictions_test.csv", index=False)
    res = {"modele": MODELE, "entraine": "mask_decoder", "meilleure_epoque": meilleure[1],
           "val": historique[meilleure[1]], "test": metriques(test.erreur_px),
           "duree_min": round((time.time() - t0) / 60, 1), "device": device, "parametres": vars(a)}
    (SORTIE / "metriques.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nmeilleure époque {meilleure[1]} ; test :")
    for k, v in res["test"].items():
        print(f"  {k:20s} {v}")


if __name__ == "__main__":
    main()
