"""Petit réseau sur les produits scalaires de la photo avec la famille de marches (une marche tous les 5 px).

Entrée : ~120 produits scalaires (photo normalisée · marche normalisée), un par position de frontière (505, 510, ..., 1095 px).
Sortie : hauteur d'eau. Même découpage train / test, mêmes photos annotées que la POD.

    .venv/bin/python marche/reseau_marches.py
"""
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir())
W0, Y0, Y1, MARGE, E = 1919, 500, 1100, 40, 4
T = ((W0 - 2 * MARGE) // E, (Y1 - Y0) // E); h = T[1]
cat = pd.read_csv(ROOT / "donnees" / "catalogue_images.csv").set_index("fichier")
split = pd.read_csv(ROOT / "donnees" / "split_train_test.csv").set_index("fichier")["split"]
ann = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "donnees" / "annotations" / "niveau_*.csv")))])
ann = ann.dropna(subset=["niveau_px"]).drop_duplicates("fichier").set_index("fichier")
hauteur = ann["niveau_px"] - cat["decalage_y"].reindex(ann.index)

def profil(nom):
    """Bande normalisée, moyennée sur chaque ligne (une marche est constante sur une ligne : seul ce profil compte)."""
    img = Image.open(ROOT / "data" / "images" / nom); img.draft("L", (img.size[0] // E, img.size[1] // E)); img = img.convert("L"); s = img.size[0] / W0
    g, t = round((MARGE + cat.at[nom, "decalage_x"]) * s), round((Y0 + cat.at[nom, "decalage_y"]) * s)
    x = np.asarray(img.crop((g, t, g + round((W0 - 2 * MARGE) * s), t + round((Y1 - Y0) * s))).resize(T), dtype=np.float64)
    return ((x - x.mean()) / x.std()).mean(axis=1)

utiles = sorted(cat.index[(cat.exploitable == "oui") & cat.decalage_y.notna()])
fich = {s: [f for f in utiles if split.get(f) == s and np.isfinite(hauteur.get(f, np.nan))] for s in ("train", "test")}
with ThreadPoolExecutor(8) as ex:
    P = {s: np.stack(list(ex.map(profil, f))) for s, f in fich.items()}
y = {s: hauteur.reindex(f).to_numpy() for s, f in fich.items()}

# famille de marches : frontière tous les 5 px (pleine résolution) -> position en lignes de la bande réduite
positions = np.arange(Y0 + 5, Y1, 5)                                  # 505, 510, ..., 1095
lignes = np.arange(h) * E + Y0 + E / 2                                # centre de chaque ligne, en px
M = np.where(lignes[None, :] >= positions[:, None], 1.0, -1.0)
M = (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
F = {s: P[s] @ M.T / h for s in P}                                    # (n, 119) produits scalaires
print(f"{len(positions)} marches | train {F['train'].shape}, test {F['test'].shape}")

def metr(p):
    e = np.abs(p - y["test"])
    return dict(moyenne=e.mean(), mediane=np.median(e), max=e.max(), a5=100 * (e <= 5).mean(), a10=100 * (e <= 10).mean(), a20=100 * (e <= 20).mean())

sx = StandardScaler().fit(F["train"]); Xtr, Xte = sx.transform(F["train"]), sx.transform(F["test"])
my, sy = y["train"].mean(), y["train"].std()
res = {"maximum du produit scalaire (sans apprentissage)": metr(positions[F["test"].argmax(1)].astype(float)),
       "régression linéaire": metr(Ridge(1.0).fit(Xtr, (y["train"] - my) / sy).predict(Xte) * sy + my)}
preds = {}
for nom, couches in [("petit réseau (16)", (16,)), ("petit réseau (32, 16)", (32, 16)), ("réseau (64, 32)", (64, 32))]:
    p = [MLPRegressor(hidden_layer_sizes=couches, alpha=1e-3, max_iter=3000, early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=50, random_state=g).fit(Xtr, (y["train"] - my) / sy).predict(Xte) * sy + my for g in (0, 1, 2)]
    m = [metr(q) for q in p]
    res[nom] = {k: np.mean([x[k] for x in m]) for k in m[0]}; preds[nom] = np.mean(p, 0)
    n_par = sum(a * b + b for a, b in zip((len(positions),) + couches, couches + (1,)))
    print(f"{nom} : {n_par} paramètres")
print(pd.DataFrame(res).T.round(1).to_string())

# figure : prédiction vs vérité pour le meilleur
best = min(preds, key=lambda k: res[k]["moyenne"]); p = preds[best]
fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
ax[0].scatter(y["test"], p, s=6, alpha=.5, color="#1d3557"); lim = [580, 900]
ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xlim(lim); ax[0].set_ylim(lim); ax[0].grid(alpha=.3)
ax[0].set_xlabel("hauteur annotée (px)"); ax[0].set_ylabel("hauteur prédite (px)")
ax[0].set_title(f"{best} sur les produits scalaires avec les marches : moyenne {res[best]['moyenne']:.1f} px", loc="left", fontsize=11)
dates = pd.to_datetime(pd.Series(fich["test"]).str[:10])
ax[1].scatter(dates, p - y["test"], s=6, alpha=.6, color="#1d3557"); ax[1].axhline(0, color="k", lw=1); ax[1].grid(alpha=.3)
ax[1].set_ylabel("prédite − vraie (px)"); ax[1].set_title("erreur au cours de la saison", loc="left")
plt.tight_layout(); fig.savefig(Path(__file__).resolve().parent / "figures" / "reseau_marches.png", dpi=110)
