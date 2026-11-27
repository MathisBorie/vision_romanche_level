"""Petit réseau convolutif entraîné depuis zéro : hauteur d'eau à partir de la bande (noir et blanc, 1/4).

Même prétraitement que la POD (notebook 06) : bande y ∈ [500, 1100] recalée, noir et blanc, réduite à 1/4
(150 × 459), normalisée photo par photo. Même découpage train / test (split_train_test.csv).
Sur le train, quelques semaines sont mises de côté (validation) pour choisir l'époque : le test n'est regardé qu'à la fin.

    .venv/bin/python cnn/cnn_bande.py --graines 0 1 2
"""
from __future__ import annotations
import argparse, glob, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir())
W0, Y0, Y1, MARGE, ECHELLE = 1919, 500, 1100, 40, 4
TAILLE = ((W0 - 2 * MARGE) // ECHELLE, (Y1 - Y0) // ECHELLE)        # (459, 150)


def charger_donnees():
    cat = pd.read_csv(ROOT / "donnees" / "catalogue_images.csv").set_index("fichier")
    sp = pd.read_csv(ROOT / "donnees" / "split_train_test.csv").set_index("fichier")
    ann = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "donnees" / "annotations" / "niveau_*.csv")))])
    ann = ann.dropna(subset=["niveau_px"]).drop_duplicates("fichier").set_index("fichier")[["niveau_px"]]
    ann = ann.join(cat[["decalage_x", "decalage_y", "exploitable"]], how="inner").join(sp[["semaine", "split"]], how="inner")
    ann = ann[(ann.exploitable == "oui") & ann.decalage_y.notna()].sort_index()
    ann["hauteur"] = ann.niveau_px - ann.decalage_y

    def bande(nom):
        img = Image.open(ROOT / "data" / "images" / nom); img.draft("L", (img.size[0] // ECHELLE, img.size[1] // ECHELLE))
        img = img.convert("L"); s = img.size[0] / W0
        g, t = round((MARGE + ann.at[nom, "decalage_x"]) * s), round((Y0 + ann.at[nom, "decalage_y"]) * s)
        return np.asarray(img.crop((g, t, g + round((W0 - 2 * MARGE) * s), t + round((Y1 - Y0) * s))).resize(TAILLE), dtype=np.uint8)
    with ThreadPoolExecutor(8) as ex:
        X = np.stack(list(ex.map(bande, ann.index)))                 # (n, 150, 459) uint8, non normalisé
    return ann, X


def normaliser(x):                                                    # x : (B, 1, H, W) float
    m = x.mean(dim=(2, 3), keepdim=True); s = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
    return (x - m) / s


def augmenter(x, gen):
    """Avant normalisation (x dans [0, 1]) : ombres locales, gamma, bruit, symétrie gauche-droite, petit décalage horizontal.
    Aucune ne change la hauteur de la frontière."""
    B, _, H, W = x.shape
    dev = x.device
    # gamma (change le contraste de façon non linéaire)
    gamma = torch.exp(torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.4, 0.4, generator=gen))
    x = x.clamp_min(1e-4) ** gamma
    # ombre douce : une grande tache elliptique plus sombre, à un endroit au hasard
    yy = torch.linspace(0, 1, H, device=dev).view(1, 1, H, 1); xx = torch.linspace(0, 1, W, device=dev).view(1, 1, 1, W)
    cy, cx = (torch.rand(B, 1, 1, 1, device=dev, generator=gen) for _ in range(2))
    ry = 0.2 + 0.6 * torch.rand(B, 1, 1, 1, device=dev, generator=gen); rx = 0.15 + 0.5 * torch.rand(B, 1, 1, 1, device=dev, generator=gen)
    tache = torch.exp(-(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2))
    force = 0.6 * torch.rand(B, 1, 1, 1, device=dev, generator=gen) * (torch.rand(B, 1, 1, 1, device=dev, generator=gen) < 0.6)
    x = x * (1 - force * tache)
    # bruit
    x = x + 0.02 * torch.randn(x.shape, device=dev, generator=gen)
    # symétrie gauche-droite (une fois sur deux)
    flip = torch.rand(B, device=dev, generator=gen) < 0.5
    x = torch.where(flip.view(B, 1, 1, 1), x.flip(3), x)
    # petit décalage horizontal (±8 px réduits)
    d = int(torch.randint(-8, 9, (1,), generator=gen, device=dev))
    return torch.roll(x, d, dims=3)


class PetitCNN(nn.Module):
    def __init__(self, c=(16, 32, 64, 64, 96)):
        super().__init__()
        couches, entree = [], 1
        for k in c:
            couches += [nn.Conv2d(entree, k, 3, padding=1, bias=False), nn.BatchNorm2d(k), nn.ReLU(inplace=True),
                        nn.Conv2d(k, k, 3, padding=1, bias=False), nn.BatchNorm2d(k), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            entree = k
        self.convs = nn.Sequential(*couches)                          # 150 x 459 -> 4 x 14
        self.tete = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(c[-1] * 4, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):
        z = self.convs(x)                 # (B, C, 4, 14)
        z = z.mean(dim=3)                 # moyenne sur la largeur : la hauteur de la frontière ne dépend pas de x
        return self.tete(z).squeeze(1)


def entrainer(X, ann, graine, epoques=40, lot=48, dev="mps"):
    torch.manual_seed(graine); gen = torch.Generator(device=dev); gen.manual_seed(graine)
    tr_sem = sorted(ann.loc[ann.split == "train", "semaine"].unique())
    val_sem = tr_sem[2::5]                                            # 1 semaine de train sur 5 pour la validation
    i_tr = np.where((ann.split == "train") & ~ann.semaine.isin(val_sem))[0]
    i_va = np.where(ann.semaine.isin(val_sem) & (ann.split == "train"))[0]
    i_te = np.where(ann.split == "test")[0]
    y = ann.hauteur.to_numpy(np.float32); my, sy = y[i_tr].mean(), y[i_tr].std()
    Xt = torch.from_numpy(X).to(dev); yt = torch.from_numpy((y - my) / sy).to(dev)

    def lot_x(idx, aug):
        x = Xt[idx].unsqueeze(1).float() / 255
        return normaliser(augmenter(x, gen) if aug else x)

    def predire(m, idx):
        m.eval(); out = []
        with torch.no_grad():
            for k in range(0, len(idx), 128):
                out.append(m(lot_x(idx[k:k + 128], False)))
        return (torch.cat(out).cpu().numpy() * sy + my)

    m = PetitCNN().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-3)
    pas = epoques * int(np.ceil(len(i_tr) / lot))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=pas)
    meilleur, etat, hist = np.inf, None, []
    for ep in range(epoques):
        m.train(); t0 = time.time(); perm = np.random.default_rng(graine * 100 + ep).permutation(i_tr)
        for k in range(0, len(perm), lot):
            idx = perm[k:k + lot]
            perte = F.smooth_l1_loss(m(lot_x(idx, True)), yt[idx], beta=0.1)
            opt.zero_grad(); perte.backward(); opt.step(); sched.step()
        e_va = np.abs(predire(m, i_va) - y[i_va]).mean()
        hist.append(e_va)
        if e_va < meilleur:
            meilleur, ep_best, etat = e_va, ep, {k: v.detach().clone() for k, v in m.state_dict().items()}
        print(f"graine {graine} époque {ep + 1:2d} : validation {e_va:5.1f} px ({time.time() - t0:.0f} s)", flush=True)
    m.load_state_dict(etat)
    return {"graine": graine, "epoque": ep_best + 1, "validation": meilleur, "pred_test": predire(m, i_te), "y_test": y[i_te],
            "fichiers_test": ann.index[i_te].tolist(), "historique": hist, "val_semaines": [int(s) for s in val_sem],
            "n": {"train": len(i_tr), "validation": len(i_va), "test": len(i_te)}, "modele": m}


if __name__ == "__main__":
    pa = argparse.ArgumentParser(); pa.add_argument("--graines", type=int, nargs="+", default=[0]); pa.add_argument("--epoques", type=int, default=40)
    a = pa.parse_args()
    t0 = time.time(); ann, X = charger_donnees(); print(f"{len(ann)} photos annotées chargées en {time.time() - t0:.0f} s, bandes {X.shape[1:]}")
    res = []
    for g in a.graines:
        r = entrainer(X, ann, g, a.epoques); res.append(r)
        e = np.abs(r["pred_test"] - r["y_test"])
        print(f"== graine {g} : époque {r['epoque']}, test : moyenne {e.mean():.1f} px, médiane {np.median(e):.1f}, max {e.max():.0f}, "
              f"≤5 {100 * (e <= 5).mean():.1f} %, ≤10 {100 * (e <= 10).mean():.1f} %, ≤20 {100 * (e <= 20).mean():.1f} %", flush=True)
        torch.save(r["modele"].state_dict(), ROOT / "data" / f"cnn_bande_graine{g}.pt")
    p = np.mean([r["pred_test"] for r in res], axis=0)
    pd.DataFrame({"fichier": res[0]["fichiers_test"], "vraie": res[0]["y_test"], "pred": p,
                  **{f"pred_g{r['graine']}": r["pred_test"] for r in res}}).to_csv(ROOT / "data" / "cnn_bande_predictions_test.csv", index=False)
    json.dump([{k: v for k, v in r.items() if k not in ("pred_test", "y_test", "fichiers_test", "modele")} | {"historique": [float(h) for h in r["historique"]], "validation": float(r["validation"])}
               for r in res], open(ROOT / "data" / "cnn_bande_resume.json", "w"), indent=1)
    e = np.abs(p - res[0]["y_test"])
    print(f"== moyenne des {len(res)} graines : moyenne {e.mean():.1f} px, médiane {np.median(e):.1f}, max {e.max():.0f}, "
          f"≤5 {100 * (e <= 5).mean():.1f} %, ≤10 {100 * (e <= 10).mean():.1f} %, ≤20 {100 * (e <= 20).mean():.1f} % | {time.time() - t0:.0f} s au total")
