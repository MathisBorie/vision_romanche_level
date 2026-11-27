"""Figures de la partie POD de la présentation (POD noir et blanc du notebook 06).

    .venv/bin/python rapport/diapos/figures_pod.py
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir())
IMAGES, OUT = ROOT / "data" / "images", Path(__file__).resolve().parent / "figures"
W0, H0, Y0, Y1, MARGE, ECHELLE = 1919, 1753, 500, 1100, 40, 4
TAILLE = ((W0 - 2 * MARGE) // ECHELLE, (Y1 - Y0) // ECHELLE)
h, w = TAILLE[1], TAILLE[0]
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
cat = pd.read_csv(ROOT / "donnees" / "catalogue_images.csv").set_index("fichier")
split = pd.read_csv(ROOT / "donnees" / "split_train_test.csv").set_index("fichier")["split"]

def decouper(img, dx=0.0, dy=0.0):
    s = img.size[0] / W0
    g, t = round((MARGE + dx) * s), round((Y0 + dy) * s)
    return img.crop((g, t, g + round((W0 - 2 * MARGE) * s), t + round((Y1 - Y0) * s)))

def charger(nom):
    img = Image.open(IMAGES / nom); img.draft("L", (img.size[0] // ECHELLE, img.size[1] // ECHELLE))
    b = decouper(img.convert("L"), cat.at[nom, "decalage_x"], cat.at[nom, "decalage_y"]).resize(TAILLE)
    x = np.asarray(b, dtype=np.float64); return ((x - x.mean()) / x.std()).astype(np.float32).ravel()

utiles = cat.index[(cat["exploitable"] == "oui") & cat["decalage_y"].notna()]
train = sorted(f for f in utiles if split.get(f) == "train")
test = sorted(f for f in utiles if split.get(f) == "test")

# 1. dimension d'une image
REF = "2026-09-05-091505.jpg" if (IMAGES / "2026-09-05-091505.jpg").exists() else train[len(train) // 2]
img = Image.open(IMAGES / REF)
dx, dy = cat.at[REF, "decalage_x"], cat.at[REF, "decalage_y"]
fig, ax = plt.subplots(figsize=(8, 7.3))
ax.imshow(img); ax.axis("off")
ax.add_patch(plt.Rectangle((MARGE + dx, Y0 + dy), W0 - 2 * MARGE, Y1 - Y0, fill=False, ec="#e63946", lw=3))
fig.savefig(OUT / "1_photo_entiere.png", dpi=150, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
decouper(img, dx, dy).save(OUT / "1_bande.png")

# 2. les images se ressemblent : 8 bandes à des dates / heures / niveaux différents
rng = np.random.default_rng(3)
pool = cat.loc[train].assign(jour=lambda d: d.index.str[:10])
choix = []
for j in sorted(pool.jour.unique())[::max(1, len(pool.jour.unique()) // 8)][:8]:
    choix.append(rng.choice(pool.index[pool.jour == j]))
etiquettes = []
for i, nom in enumerate(choix):
    decouper(Image.open(IMAGES / nom), cat.at[nom, "decalage_x"], cat.at[nom, "decalage_y"]).resize((920, 300)).save(OUT / f"2_bande_{i}.png")
    etiquettes.append(f"{nom[8:10]}/{nom[5:7]} à {nom[11:13]}h{nom[13:15]}")
open(OUT / "2_etiquettes.txt", "w").write("\n".join(etiquettes))

# 3. POD
with ThreadPoolExecutor(8) as ex:
    X = np.stack(list(ex.map(charger, train)))
xbar = X.mean(axis=0); A = X - xbar
val, vec = np.linalg.eigh((A @ A.T).astype(np.float64))
o = np.argsort(val)[::-1]; val, vec = np.clip(val[o], 0, None)[:-1], vec[:, o][:, :-1]
sig = np.sqrt(val); energie = val / val.sum(); KMAX = len(val)
mode = lambda k: (A.T @ vec[:, k - 1].astype(np.float32)) / np.float32(sig[k - 1])
print(len(train), "photos train,", KMAX, "modes")

fig, ax = plt.subplots(figsize=(9, 5.5))
K = np.arange(1, KMAX + 1)
ax.plot(K, 100 * np.cumsum(energie), color="#1d3557", lw=3)
for k, c in [(50, "#457b9d"), (750, "#e63946")]:
    v = 100 * energie[:k].sum()
    ax.plot([k], [v], "o", ms=10, color=c); ax.annotate(f"{k} modes : {v:.0f} %", (k, v), (k * 1.6, v - 9), fontsize=15, color=c,
                                                       arrowprops=dict(arrowstyle="-", color=c))
ax.set_xscale("log"); ax.set_ylim(0, 102); ax.set_xlim(1, KMAX)
ax.set_xlabel("nombre de modes gardés N"); ax.set_ylabel("variance expliquée (%)"); ax.grid(alpha=.3, which="both")
plt.tight_layout(); fig.savefig(OUT / "3_variance_expliquee.png", dpi=150); plt.close(fig)

def en_image(v, diverge=False):
    v = v.reshape(h, w)
    if diverge:
        m = np.abs(v).max(); rgba = plt.get_cmap("RdBu_r")((v / m + 1) / 2)
    else:
        rgba = plt.get_cmap("gray")(np.clip(v / 6 + 0.5, 0, 1))
    return Image.fromarray((rgba[..., :3] * 255).astype(np.uint8)).resize((w * 2, h * 2), Image.BICUBIC)
en_image(xbar).save(OUT / "3_moyenne.png")
MODES_MONTRES = list(range(1, 16)) + [25, 50, 100, 200, 400, 800, 1600, 3200]
for k in MODES_MONTRES:
    en_image(mode(k), diverge=True).save(OUT / f"3_mode_{k}.png")
open(OUT / "3_energie_modes.txt", "w").write("\n".join(f"{k}={100 * energie[k - 1]:.3g}" for k in MODES_MONTRES))

# 4. reconstruction d'une photo test tirée au hasard
# photo test : niveau d'eau moyen et erreur de reconstruction finale (tous les modes) autour de 20-25 %
ann = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "donnees" / "annotations").glob("niveau_*.csv"))])
ann = ann.dropna(subset=["niveau_px"]).drop_duplicates("fichier").set_index("fichier")
haut = (ann["niveau_px"] - cat["decalage_y"].reindex(ann.index)).reindex(test)
cand = [f for f in test if 780 <= haut.get(f, np.nan) <= 830 and 9 <= int(f[11:13]) <= 16]
with ThreadPoolExecutor(8) as ex:
    Xc = np.stack(list(ex.map(charger, cand)))
Dc = Xc - xbar
ac = (Dc @ A.T).astype(np.float64) @ vec / sig
err_tous = np.sqrt(np.clip(np.sum(Dc.astype(np.float64) ** 2, 1) - np.sum(ac ** 2, 1), 0, None) / np.sum(Dc.astype(np.float64) ** 2, 1))
ok = np.where((err_tous >= 0.20) & (err_tous <= 0.25))[0]
print(len(cand), "photos test de niveau moyen ;", len(ok), "avec 20-25 % d'erreur finale")
nom = cand[int(rng.choice(ok))]
print("photo choisie :", nom, "hauteur", haut[nom])
x = charger(nom); a = (x - xbar) @ A.T @ vec / sig
nx = np.linalg.norm(x - xbar)
Image.open(IMAGES / nom).save(OUT / "4_photo_test.jpg", quality=90)
decouper(Image.open(IMAGES / nom), cat.at[nom, "decalage_x"], cat.at[nom, "decalage_y"]).save(OUT / "4_bande_test.png")
en_image(x).save(OUT / "4_test_pretraitee.png")
lignes = []
for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, KMAX]:
    rec = xbar + A.T @ (vec[:, :k] @ (a[:k] / sig[:k])).astype(np.float32)
    err = np.linalg.norm(x - rec) / nx
    en_image(rec).save(OUT / f"4_reconstruction_{k:04d}.png")
    lignes.append((k, 100 * energie[:k].sum(), 100 * err))
pd.DataFrame(lignes, columns=["K", "variance", "erreur"]).to_csv(OUT / "4_reconstruction.csv", index=False)
open(OUT / "infos.txt", "w").write(f"photo_ref={REF}\nphoto_test={nom}\nn_train={len(train)}\nkmax={KMAX}\n"
                                    f"variance_50={100*energie[:50].sum():.1f}\nvariance_750={100*energie[:750].sum():.1f}\n"
                                    f"mode1={100*energie[0]:.1f}\nmode2={100*energie[1]:.1f}\nmode3={100*energie[2]:.1f}\n")
print(open(OUT / "infos.txt").read()); print(pd.read_csv(OUT / "4_reconstruction.csv").round(1).to_string(index=False))
