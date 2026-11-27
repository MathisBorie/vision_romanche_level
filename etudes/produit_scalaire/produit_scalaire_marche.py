"""Famille d'images « marche » (noir en haut, blanc en bas, frontière à la ligne y) et produit scalaire avec les bandes normalisées.

Pour chaque photo (bande recalée, noir et blanc, 1/4, normalisée) et chaque position y de la frontière de la marche,
on calcule le produit scalaire / nombre de pixels = corrélation entre la photo et la marche (marche normalisée elle aussi).
Si l'eau est plus claire que la berge, la corrélation devrait être maximale quand la marche tombe sur la vraie frontière.
Aucun apprentissage : la position du maximum est directement une estimation de la hauteur.

    .venv/bin/python marche/produit_scalaire_marche.py
"""
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir())
OUT = Path(__file__).resolve().parent / "figures"; OUT.mkdir(parents=True, exist_ok=True)
W0, Y0, Y1, MARGE, E = 1919, 500, 1100, 40, 4
T = ((W0 - 2 * MARGE) // E, (Y1 - Y0) // E); h, w = T[1], T[0]
cat = pd.read_csv(ROOT / "donnees" / "catalogue_images.csv").set_index("fichier")
split = pd.read_csv(ROOT / "donnees" / "split_train_test.csv").set_index("fichier")["split"]
ann = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "donnees" / "annotations" / "niveau_*.csv")))])
ann = ann.dropna(subset=["niveau_px"]).drop_duplicates("fichier").set_index("fichier")
hauteur = ann["niveau_px"] - cat["decalage_y"].reindex(ann.index)

def charger(nom):
    img = Image.open(ROOT / "data" / "images" / nom); img.draft("L", (img.size[0] // E, img.size[1] // E)); img = img.convert("L"); s = img.size[0] / W0
    g, t = round((MARGE + cat.at[nom, "decalage_x"]) * s), round((Y0 + cat.at[nom, "decalage_y"]) * s)
    x = np.asarray(img.crop((g, t, g + round((W0 - 2 * MARGE) * s), t + round((Y1 - Y0) * s))).resize(T), dtype=np.float64)
    return (x - x.mean()) / x.std()

test = [f for f in sorted(cat.index[(cat.exploitable == "oui") & cat.decalage_y.notna()]) if split.get(f) == "test" and np.isfinite(hauteur.get(f, np.nan))]
with ThreadPoolExecutor(8) as ex:
    X = np.stack(list(ex.map(charger, test)))                         # (n, h, w)
y_vrai = hauteur.reindex(test).to_numpy()

# famille de marches : frontière entre la ligne y-1 et la ligne y (y = 1 .. h-1), normalisées (moyenne 0, écart type 1)
ys = np.arange(1, h)
marches = np.where(np.arange(h)[None, :] >= ys[:, None], 1.0, -1.0)  # (len(ys), h) : -1 en haut (noir), +1 en bas (blanc)
marches = (marches - marches.mean(1, keepdims=True)) / marches.std(1, keepdims=True)
profil = X.mean(axis=2)                                               # une marche est constante sur une ligne : le produit scalaire
ps = profil @ marches.T / h                                           # ne dépend que du profil moyen ligne par ligne -> (n, len(ys))
hauteur_ligne = lambda yl: Y0 + E * yl                                # ligne de la bande réduite -> hauteur (px pleine résolution)
y_pred = hauteur_ligne(ys[ps.argmax(1)])
err = y_pred - y_vrai
print(f"{len(test)} photos test | erreur du maximum du produit scalaire : moyenne {np.abs(err).mean():.1f} px, "
      f"médiane {np.median(np.abs(err)):.1f} px, à 10 px {100 * (np.abs(err) <= 10).mean():.0f} %, à 20 px {100 * (np.abs(err) <= 20).mean():.0f} %")

# 1. la famille de marches
fig, axes = plt.subplots(1, 6, figsize=(18, 2.2))
for ax, yl in zip(axes, [15, 40, 65, 90, 115, 140]):
    ax.imshow(np.where(np.arange(h)[:, None] >= yl, 1, -1) * np.ones((1, w)), cmap="gray", vmin=-1.2, vmax=1.2)
    ax.set_title(f"frontière à {hauteur_ligne(yl)} px", fontsize=11); ax.axis("off")
plt.tight_layout(); fig.savefig(OUT / "famille_marches.png", dpi=110); plt.close(fig)

# 2. exemples : bande + courbe du produit scalaire
rng = np.random.default_rng(1)
ex_idx = sorted(rng.choice(len(test), 8, replace=False), key=lambda i: y_vrai[i])
fig, axes = plt.subplots(len(ex_idx), 2, figsize=(17, 2.6 * len(ex_idx)), gridspec_kw={"width_ratios": [1.6, 1]})
for (a0, a1), i in zip(axes, ex_idx):
    a0.imshow(X[i], cmap="gray", vmin=-3, vmax=3, extent=[0, w, hauteur_ligne(h), Y0], aspect="auto")
    a0.axhline(y_vrai[i], color="lime", lw=2); a0.axhline(y_pred[i], color="red", lw=1.5, ls="--"); a0.axis("off")
    a0.set_title(f"{test[i][:10]} à {test[i][11:13]}h{test[i][13:15]} : annoté {y_vrai[i]:.0f} px (vert), maximum {y_pred[i]:.0f} px (rouge)", loc="left", fontsize=11)
    a1.plot(hauteur_ligne(ys), ps[i], color="#1d3557", lw=2)
    a1.axvline(y_vrai[i], color="lime", lw=2.5, label="frontière annotée"); a1.axvline(y_pred[i], color="red", ls="--", label="maximum")
    a1.set_xlim(Y0, hauteur_ligne(h)); a1.grid(alpha=.3); a1.set_ylabel("produit scalaire")
axes[0, 1].legend(fontsize=9); axes[-1, 1].set_xlabel("position de la frontière de la marche (px, comme la hauteur d'eau)")
plt.tight_layout(); fig.savefig(OUT / "exemples_produit_scalaire.png", dpi=100); plt.close(fig)

# 3. toutes les photos test : courbes recentrées sur la frontière annotée
fig, ax = plt.subplots(1, 2, figsize=(17, 5.5))
dec = np.arange(-200, 201, 4)
courbes = np.array([np.interp(y_vrai[i] + dec, hauteur_ligne(ys), ps[i], left=np.nan, right=np.nan) for i in range(len(test))])
for c in courbes[rng.choice(len(test), 120, replace=False)]:
    ax[0].plot(dec, c, color="#1d3557", alpha=.12, lw=1)
ax[0].plot(dec, np.nanmedian(courbes, 0), color="#e63946", lw=3, label="médiane des 883 photos test")
ax[0].axvline(0, color="lime", lw=2.5, label="frontière annotée"); ax[0].grid(alpha=.3); ax[0].legend()
ax[0].set_xlabel("position de la marche − frontière annotée (px)"); ax[0].set_ylabel("produit scalaire")
ax[0].set_title("Produit scalaire autour de la vraie frontière", loc="left")
ax[1].scatter(y_vrai, y_pred, s=6, alpha=.5, color="#1d3557"); lim = [Y0, hauteur_ligne(h)]
ax[1].plot(lim, lim, "k--", lw=1); ax[1].set_xlim(580, 900); ax[1].set_ylim(Y0, hauteur_ligne(h)); ax[1].grid(alpha=.3)
ax[1].set_xlabel("hauteur annotée (px)"); ax[1].set_ylabel("position du maximum (px)")
ax[1].set_title(f"Maximum du produit scalaire comme estimation : erreur moyenne {np.abs(err).mean():.0f} px, médiane {np.median(np.abs(err)):.0f} px",
                loc="left", fontsize=11)
plt.tight_layout(); fig.savefig(OUT / "toutes_photos_test.png", dpi=110); plt.close(fig)
