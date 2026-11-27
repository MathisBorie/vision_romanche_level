"""Diapo « répartition de l'erreur » : POD + réseau (K = 750), photos test.

    .venv/bin/python presentation/figures_erreur.py
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir()); OUT = Path(__file__).resolve().parent / "figures"
W0, Y0, Y1, MARGE, K = 1919, 500, 1100, 40, 750
d = np.load(ROOT / "data" / "pod_bande.npz")
cat = pd.read_csv(ROOT / "donnees" / "catalogue_images.csv").set_index("fichier")
ann = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "donnees" / "annotations" / "niveau_*.csv")))])
ann = ann.dropna(subset=["niveau_px"]).drop_duplicates("fichier").set_index("fichier")
haut = ann["niveau_px"] - cat["decalage_y"].reindex(ann.index)
def jeu(s):
    f = d[f"fichiers_{s}"]; y = haut.reindex(f).to_numpy(); g = np.isfinite(y)
    return d[f"coefs_{s}"][g, :K], y[g], f[g]
Ctr, ytr, _ = jeu("train"); Cte, yte, noms = jeu("test")
sx = StandardScaler().fit(Ctr); my, sy = ytr.mean(), ytr.std()
pred = np.mean([MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=3000, early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=50, random_state=g).fit(sx.transform(Ctr), (ytr - my) / sy).predict(sx.transform(Cte)) * sy + my
                for g in (0, 1, 2)], axis=0)
err = pred - yte; ae = np.abs(err)
stats = dict(moyenne=ae.mean(), mediane=np.median(ae), q90=np.quantile(ae, .9), a10=100 * (ae <= 10).mean(), a20=100 * (ae <= 20).mean(), biais=err.mean())
print({k: round(v, 1) for k, v in stats.items()})

BLEU, ROUGE, GRIS = "#1d3557", "#e63946", "#6c757d"
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8, 5.6))
bins = np.arange(0, 62, 2)
ax.hist(np.clip(ae, 0, 60), bins=bins, color=BLEU, alpha=.85)
ax.axvline(stats["mediane"], color=ROUGE, lw=2.5, ls="--"); ax.axvline(stats["q90"], color=GRIS, lw=2.5, ls="--")
ym = ax.get_ylim()[1]
ax.text(31, ym * 0.62, f"moyenne : {stats['moyenne']:.1f} px".replace(".", ","), fontsize=15, color=BLEU, fontweight="bold")
ax.text(31, ym * 0.52, f"médiane : {stats['mediane']:.0f} px", fontsize=15, color=ROUGE, fontweight="bold")
ax.text(31, ym * 0.42, f"90 % des photos < {stats['q90']:.0f} px", fontsize=15, color=GRIS, fontweight="bold")
ax.set_xlim(0, 60); ax.set_xlabel("|erreur| sur la hauteur d'eau (px)"); ax.set_ylabel("nombre de photos test")
plt.tight_layout(); fig.savefig(OUT / "6_repartition_erreur.png", dpi=150); plt.close(fig)

# photo de jour avec une erreur d'environ 10 px
heure = np.array([int(n[11:13]) for n in noms])
cand = np.where((np.abs(ae - 10) <= 0.5) & (heure >= 10) & (heure <= 16))[0]
i = int(np.random.default_rng(2).choice(cand)); nom = noms[i]
dx, dy = cat.at[nom, "decalage_x"], cat.at[nom, "decalage_y"]
img = Image.open(ROOT / "data" / "images" / nom).convert("RGB")
s = img.size[0] / W0
bande = img.crop((round((MARGE + dx) * s), round((Y0 + dy) * s), round((W0 - MARGE + dx) * s), round((Y1 + dy) * s)))
def lignes(im, y_off, ep):
    dr = ImageDraw.Draw(im)
    for yv, c in ((yte[i], (0, 230, 0)), (pred[i], (230, 30, 40))):
        y = round((yv - y_off) * s); dr.line([(0, y), (im.size[0], y)], fill=c, width=ep)
    return im
lignes(bande.copy(), Y0, 7).save(OUT / "6_exemple_10px_bande.png")
# zoom : 120 px de haut autour de la frontière, sur un tiers de la largeur
yc = (yte[i] + pred[i]) / 2
zoom = img.crop((round((MARGE + dx + 600) * s), round((yc - 60 + dy) * s), round((MARGE + dx + 1200) * s), round((yc + 60 + dy) * s)))
zoom = lignes(zoom, yc - 60, 2).resize((zoom.size[0] * 2, zoom.size[1] * 2), Image.NEAREST)
zoom.save(OUT / "6_exemple_10px_zoom.png")
(OUT / "6_infos.txt").write_text("\n".join([f"{k}={v:.1f}" for k, v in stats.items()] +
                                           [f"photo={nom}", f"vraie={yte[i]:.0f}", f"pred={pred[i]:.0f}", f"erreur={err[i]:+.0f}"]))
print(nom, yte[i], pred[i])

# les trois photos des diapos de reconstruction : erreur du réseau et bande avec les deux lignes
TROIS = [("eau haute", "2026-06-24-151505.jpg"), ("niveau moyen", (OUT / "infos.txt").read_text().split("photo_test=")[1].split()[0]),
         ("eau basse", "2026-10-06-141511.jpg")]
lignes_txt = []
for j, (et, n) in enumerate(TROIS):
    k = int(np.where(noms == n)[0][0])
    dx, dy = cat.at[n, "decalage_x"], cat.at[n, "decalage_y"]
    im = Image.open(ROOT / "data" / "images" / n).convert("RGB"); s = im.size[0] / W0
    b = im.crop((round((MARGE + dx) * s), round((Y0 + dy) * s), round((W0 - MARGE + dx) * s), round((Y1 + dy) * s)))
    dr = ImageDraw.Draw(b)
    for yv, c in ((yte[k], (0, 230, 0)), (pred[k], (230, 30, 40))):
        yy = round((yv - Y0) * s); dr.line([(0, yy), (b.size[0], yy)], fill=c, width=7)
    b.save(OUT / f"6_trois_{j}.png")
    lignes_txt.append(f"{j}|{et}|{n}|{yte[k]:.0f}|{pred[k]:.0f}|{pred[k] - yte[k]:+.1f}")
    print(et, n, "annoté", round(yte[k]), "prédit", round(pred[k]), "erreur", round(pred[k] - yte[k], 1))
(OUT / "6_trois.txt").write_text("\n".join(lignes_txt))
