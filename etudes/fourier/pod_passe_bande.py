"""POD sur les bandes filtrées par un passe-bande de Fourier, puis même réseau que le notebook 06.

Chaque bande (recalée, noir et blanc, 1/4 : 150 x 459) est filtrée dans le domaine de Fourier par un passe-bande
gaussien (on coupe les très basses fréquences = éclairage et grandes ombres, et les très hautes = texture fine de l'eau),
puis normalisée photo par photo. POD sur le train, réseau (64, 32) sur les K premiers coefficients, 3 graines.

    .venv/bin/python fourier/pod_passe_bande.py
"""
import glob, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir()); OUT = Path(__file__).resolve().parent / "figures"; OUT.mkdir(parents=True, exist_ok=True)
W0, Y0, Y1, MARGE, E = 1919, 500, 1100, 40, 4
T = ((W0 - 2 * MARGE) // E, (Y1 - Y0) // E); h, w = T[1], T[0]
cat = pd.read_csv(ROOT / "donnees" / "catalogue_images.csv").set_index("fichier")
split = pd.read_csv(ROOT / "donnees" / "split_train_test.csv").set_index("fichier")["split"]
ann = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "donnees" / "annotations" / "niveau_*.csv")))])
ann = ann.dropna(subset=["niveau_px"]).drop_duplicates("fichier").set_index("fichier")
haut = ann["niveau_px"] - cat["decalage_y"].reindex(ann.index)

def bande(nom):
    img = Image.open(ROOT / "data" / "images" / nom); img.draft("L", (img.size[0] // E, img.size[1] // E)); img = img.convert("L"); s = img.size[0] / W0
    g, t = round((MARGE + cat.at[nom, "decalage_x"]) * s), round((Y0 + cat.at[nom, "decalage_y"]) * s)
    return np.asarray(img.crop((g, t, g + round((W0 - 2 * MARGE) * s), t + round((Y1 - Y0) * s))).resize(T), dtype=np.float32)

utiles = sorted(cat.index[(cat.exploitable == "oui") & cat.decalage_y.notna()])
fich = {s: [f for f in utiles if split.get(f) == s] for s in ("train", "test")}
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    B = {s: np.stack(list(ex.map(bande, f))) for s, f in fich.items()}
print({s: b.shape for s, b in B.items()}, f"{time.time() - t0:.0f} s", flush=True)

# fréquences (cycles par pixel de la bande réduite ; 1 pixel réduit = 4 px de la photo)
fy, fx = np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None, :]
fr = np.sqrt(fy ** 2 + fx ** 2)
def passe_bande(p_max, p_min):
    """Gaussien : garde les détails de taille entre p_min et p_max pixels réduits (période)."""
    bas = np.exp(-(fr * p_min) ** 2 * 2)                      # coupe les hautes fréquences (détails < p_min)
    haut_ = 1 - np.exp(-(fr * p_max) ** 2 * 2) if p_max else 1  # coupe les basses fréquences (structures > p_max)
    return (bas * haut_).astype(np.float32)
FILTRES = {"sans filtre": None,
           "passe-haut (coupe > 40 px réduits = 160 px)": passe_bande(40, 0),
           "passe-bande (de 8 à 160 px)": passe_bande(40, 2),
           "passe-bande étroit (de 16 à 80 px)": passe_bande(20, 4)}

def filtrer(X, H):
    if H is None:
        Y = X.copy()
    else:
        Y = np.fft.irfft2(np.fft.rfft2(X, axes=(1, 2)) * H, s=(h, w), axes=(1, 2)).astype(np.float32)
    Y -= Y.mean(axis=(1, 2), keepdims=True); Y /= Y.std(axis=(1, 2), keepdims=True)
    return Y.reshape(len(Y), -1)

def jeu(f, C):
    y = haut.reindex(f).to_numpy(); g = np.isfinite(y); return C[g], y[g]

KS = [10, 50, 100, 300, 750, 1000, 2000]
res, exemples = {}, {}
for nom, H in FILTRES.items():
    t0 = time.time()
    X = {s: filtrer(B[s], H) for s in B}
    exemples[nom] = X["test"][100].reshape(h, w)
    xbar = X["train"].mean(0); A = X["train"] - xbar
    val, vec = np.linalg.eigh((A @ A.T).astype(np.float64)); o = np.argsort(val)[::-1]
    val, vec = np.clip(val[o], 0, None)[:-1], vec[:, o][:, :-1]; sig = np.sqrt(val); en = val / val.sum()
    C = {"train": vec * sig, "test": ((X["test"] - xbar) @ A.T).astype(np.float64) @ vec / sig}
    Ctr, ytr = jeu(fich["train"], C["train"]); Cte, yte = jeu(fich["test"], C["test"])
    ligne = {"variance 50 modes (%)": 100 * en[:50].sum()}
    for K in KS:
        sx = StandardScaler().fit(Ctr[:, :K]); my, sy = ytr.mean(), ytr.std()
        p = [MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=3000, early_stopping=True, validation_fraction=0.1,
                          n_iter_no_change=50, random_state=g).fit(sx.transform(Ctr[:, :K]), (ytr - my) / sy).predict(sx.transform(Cte[:, :K])) * sy + my
             for g in (0, 1, 2)]
        ligne[K] = np.mean([np.abs(q - yte).mean() for q in p])
        if K == 750:
            e = np.abs(np.mean(p, 0) - yte); ligne["K=750 médiane"] = np.median(e); ligne["K=750 max"] = e.max(); ligne["K=750 à 10 px (%)"] = 100 * (e <= 10).mean()
    res[nom] = ligne
    print(nom, {k: round(v, 1) for k, v in ligne.items()}, f"{time.time() - t0:.0f} s", flush=True)
    if H is not None:
        mods = ((A.T @ vec[:, :4].astype(np.float32)) / sig[:4].astype(np.float32)).T
        np.save(OUT / f"modes_{list(FILTRES).index(nom)}.npy", mods)
pd.DataFrame(res).T.to_csv(OUT / "resultats.csv")

fig, axes = plt.subplots(len(exemples), 1, figsize=(10, 2.6 * len(exemples)))
for ax, (nom, x) in zip(axes, exemples.items()):
    ax.imshow(x, cmap="gray", vmin=-3, vmax=3); ax.set_title(nom, loc="left"); ax.axis("off")
plt.tight_layout(); fig.savefig(OUT / "exemples_filtres.png", dpi=100)
