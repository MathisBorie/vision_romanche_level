"""PLS (moindres carrés partiels) sur les bandes prétraitées, à comparer à la POD du notebook 06.

Même prétraitement (bande y ∈ [500, 1100] recalée, noir et blanc, 1/4, normalisée), mêmes photos annotées train / test.
Deux façons de prédire la hauteur avec K composantes PLS :
  - PLS seule (régression linéaire, c'est la définition de la PLS),
  - PLS + réseau : les K scores PLS donnés au même réseau (64, 32) que pour la POD, 3 graines.

    .venv/bin/python pls/pls_bande.py
"""
import glob, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir())
W0, Y0, Y1, MARGE, ECHELLE = 1919, 500, 1100, 40, 4
TAILLE = ((W0 - 2 * MARGE) // ECHELLE, (Y1 - Y0) // ECHELLE)
cat = pd.read_csv(ROOT / "donnees" / "catalogue_images.csv").set_index("fichier")
split = pd.read_csv(ROOT / "donnees" / "split_train_test.csv").set_index("fichier")["split"]
ann = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "donnees" / "annotations" / "niveau_*.csv")))])
ann = ann.dropna(subset=["niveau_px"]).drop_duplicates("fichier").set_index("fichier")
hauteur = ann["niveau_px"] - cat["decalage_y"].reindex(ann.index)

def charger(nom):
    img = Image.open(ROOT / "data" / "images" / nom); img.draft("L", (img.size[0] // ECHELLE, img.size[1] // ECHELLE))
    img = img.convert("L"); s = img.size[0] / W0
    g, t = round((MARGE + cat.at[nom, "decalage_x"]) * s), round((Y0 + cat.at[nom, "decalage_y"]) * s)
    x = np.asarray(img.crop((g, t, g + round((W0 - 2 * MARGE) * s), t + round((Y1 - Y0) * s))).resize(TAILLE), dtype=np.float64)
    return ((x - x.mean()) / x.std()).astype(np.float32).ravel()

utiles = cat.index[(cat["exploitable"] == "oui") & cat["decalage_y"].notna()]
fich = {s: [f for f in sorted(utiles) if split.get(f) == s and np.isfinite(hauteur.get(f, np.nan))] for s in ("train", "test")}
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    X = {s: np.stack(list(ex.map(charger, f))) for s, f in fich.items()}
y = {s: hauteur.reindex(f).to_numpy() for s, f in fich.items()}
print({s: x.shape for s, x in X.items()}, f"chargement {time.time() - t0:.0f} s", flush=True)

K_MAX = 500
t0 = time.time()
pls = PLSRegression(n_components=K_MAX, scale=False, max_iter=500).fit(X["train"], y["train"])
print(f"PLS {K_MAX} composantes : {time.time() - t0:.0f} s", flush=True)
xm = pls._x_mean
T = {s: (X[s] - xm) @ pls.x_rotations_ for s in X}                  # scores PLS (n, K_MAX)
ym = y["train"].mean()

def metriques(p, v):
    e = np.abs(p - v)
    return dict(moyenne=e.mean(), mediane=np.median(e), max=e.max(), a5=100 * (e <= 5).mean(), a10=100 * (e <= 10).mean(), a20=100 * (e <= 20).mean())

KS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500]
res, preds = [], {}
for K in KS:
    # PLS seule : y ≈ ym + T[:, :K] q[:K]
    p_lin = ym + T["test"][:, :K] @ pls.y_loadings_[0, :K]
    # PLS + réseau
    sx = StandardScaler().fit(T["train"][:, :K]); sy = y["train"].std()
    pr = [MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=3000, early_stopping=True, validation_fraction=0.1,
                       n_iter_no_change=50, random_state=g).fit(sx.transform(T["train"][:, :K]), (y["train"] - ym) / sy)
          .predict(sx.transform(T["test"][:, :K])) * sy + ym for g in (0, 1, 2)]
    m_lin = metriques(p_lin, y["test"]); m_res = [metriques(p, y["test"]) for p in pr]
    ligne = {"K": K, **{f"pls_{k}": v for k, v in m_lin.items()},
             **{f"pls_reseau_{k}": np.mean([m[k] for m in m_res]) for k in m_lin}}
    res.append(ligne); preds[K] = (p_lin, np.mean(pr, axis=0))
    print(f"K = {K:4d} | PLS seule {m_lin['moyenne']:5.1f} px | PLS + réseau {ligne['pls_reseau_moyenne']:5.1f} px", flush=True)
pd.DataFrame(res).to_csv(ROOT / "data" / "pls_resultats.csv", index=False)
Kb = int(pd.DataFrame(res).set_index("K")["pls_moyenne"].idxmin())
pd.DataFrame({"fichier": fich["test"], "vraie": y["test"], "pls": preds[Kb][0], "pls_reseau": preds[Kb][1]}).to_csv(
    ROOT / "data" / "pls_predictions_test.csv", index=False)
np.save(ROOT / "data" / "pls_directions_10.npy", pls.x_rotations_[:, :10].T.astype(np.float32))
print("meilleur K (PLS seule) :", Kb)
