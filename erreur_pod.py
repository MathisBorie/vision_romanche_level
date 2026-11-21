"""Erreur de reconstruction de la POD (couleur et noir et blanc) en fonction du nombre de modes, moyennée sur N_TEST images.

Deux courbes :
  - images dans la base : les 100 images font partie des images ayant servi à calculer la POD ;
  - images hors base    : POD recalculée sans ces 100 images, puis projection (vraie capacité de généralisation).
Usage : python erreur_pod.py   -> data/preview/erreur_pod.png
"""
import csv
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyse_spectrale import PREVIEW, load_band
from bucket import ROOT

N_TEST = 500                               # images sur lesquelles on moyenne
KS = list(range(1, 501))                   # tous les K (le calcul est quasi gratuit, voir errors)


def pod(A, nm):
    """Modes POD (méthode des snapshots) d'une matrice centrée A (n, p) -> (nm, p) orthonormés."""
    lam, V = np.linalg.eigh((A @ A.T).astype(np.float64))
    lam, V = lam[::-1][:nm], V[:, ::-1][:, :nm]
    return ((A.T @ V.astype(np.float32)) / np.sqrt(lam).astype(np.float32)).T


def errors(B, mean, X):
    """Erreur relative moyenne ||résidu|| / ||x - moyenne|| pour chaque K de KS."""
    D = X - mean
    a = D @ B.T                                     # coefficients de projection (n, nm)
    d2 = (D.astype(np.float64) ** 2).sum(axis=1)
    cum = np.cumsum(a.astype(np.float64) ** 2, axis=1)
    # base orthonormée : ||résidu||² = ||x - moyenne||² - somme des a_k² (k <= K)
    return np.array([np.mean(np.sqrt(np.clip(d2 - cum[:, K - 1], 0, None) / d2)) for K in KS])


def curves(day, gray):
    with ThreadPoolExecutor(8) as ex:
        X = np.stack(list(ex.map(lambda r: load_band(r["key"], 8, gray=gray), day))).reshape(len(day), -1)
    test = np.round(np.linspace(0, len(day) - 1, N_TEST)).astype(int)   # réparties dans le temps
    train = np.setdiff1d(np.arange(len(day)), test)
    m_all = X.mean(axis=0)
    e_in = errors(pod(X - m_all, max(KS)), m_all, X[test])
    m_tr = X[train].mean(axis=0)
    e_out = errors(pod(X[train] - m_tr, max(KS)), m_tr, X[test])
    return e_in, e_out


def main():
    day = sorted([r for r in csv.DictReader(open(ROOT / "annotations" / "qualite.csv"))
                  if r["categorie"] != "noire" and "_permission" not in r["key"]], key=lambda r: r["datetime"])
    res = {"couleur": curves(day, False), "noir et blanc": curves(day, True)}

    K = np.array(KS)
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    for (name, (e_in, e_out)), col in zip(res.items(), ["C0", "C7"]):
        for e, ls, lab in ((e_in, "-", "images dans la base POD"), (e_out, "--", "images hors base (POD sans elles)")):
            ax[0].plot(K, 100 * e, ls, color=col, label=f"{name} : {lab}")
            # dérivée discrète dE/dK (en points de % par mode) ; elle est négative, on trace -dE/dK
            ax[1].plot(K[1:], -100 * np.diff(e), ls, color=col, lw=.8, label=f"{name} : {lab}")
    ax[0].set_xlabel("nombre de modes K"); ax[0].set_ylabel("erreur moyenne (%)  = ||résidu|| / ||image - moyenne||")
    ax[0].set_title(f"Erreur de reconstruction moyenne sur {N_TEST} images ({len(day)} images de jour)", loc="left")
    ax[0].set_ylim(0, 100)
    ax[1].set_yscale("log"); ax[1].set_xlabel("nombre de modes K")
    ax[1].set_ylabel("-dE/dK : gain apporté par le K-ième mode (points de % par mode)")
    ax[1].set_title("Dérivée de l'erreur (échelle log)", loc="left")
    for a in ax:
        a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    plt.savefig(PREVIEW / "erreur_pod.png", dpi=80, bbox_inches="tight")
    (ci, co), (gi, go) = res["couleur"], res["noir et blanc"]
    print("  K | erreur hors base : couleur  N&B | gain du K-ième mode hors base (points %) : couleur  N&B")
    for k in (2, 5, 10, 20, 50, 100, 200, 300, 400, 500):
        i = k - 1
        print(f"{k:3d} | {100 * co[i]:22.1f} {100 * go[i]:5.1f} | "
              f"{100 * (co[i - 1] - co[i]):45.4f} {100 * (go[i - 1] - go[i]):7.4f}")

if __name__ == "__main__":
    main()
