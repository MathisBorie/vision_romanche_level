"""POD des bandes + régression linéaire / réseau sur les coefficients (repris du notebook 06 de Mathis).

1. Prétraitement : bande y ∈ [500, 1100] recalée sur le mouvement de la caméra (decalage_x / decalage_y),
   marge de 40 px de chaque côté, noir et blanc, centrée réduite, réduite à 1/4 (150 x 459 px).
2. POD (méthode des snapshots) sur toutes les photos exploitables du train (annotées ou non).
3. Sur les photos annotées du train (+ val) : hauteur = niveau_px - decalage_y prédite à partir des K premiers
   coefficients. Ridge (K = 750) et perceptron 64-32 (K = 300 et 750, 3 graines, prédictions moyennées).
   K est fixé d'avance (pas choisi sur le test).

Durée d'entraînement = lecture des photos train + POD + ajustement du modèle.
Temps par image = lecture / prétraitement + projection sur les K modes + prédiction (un cœur).

    .venv/bin/python comparaison/methode_pod.py
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from commun import ROOT, Chrono, donnees, enregistrer, nom_cpu

IMAGES = ROOT / "data" / "images"
TRAITEMENT = ROOT / "donnees"
CATEGORIE = "POD"
W0 = 1919
Y0, Y1, MARGE, ECHELLE = 500, 1100, 40, 4
TAILLE = ((W0 - 2 * MARGE) // ECHELLE, (Y1 - Y0) // ECHELLE)     # (largeur, hauteur) = (459, 150)
K_MAX = 750
GRAINES = (0, 1, 2)
MODELES = {"pod_lineaire_750": ("POD + linéaire, K = 750", "ridge", 750),
           "pod_reseau_300": ("POD + réseau, K = 300", "mlp", 300),
           "pod_reseau_750": ("POD + réseau, K = 750", "mlp", 750)}


def charger(nom: str, dx: float, dy: float) -> np.ndarray:
    """Bande recalée, noir et blanc, réduite à 1/4, centrée réduite, à plat (notebook 06 : decouper + normaliser)."""
    img = Image.open(IMAGES / nom)
    img.draft("L", (img.size[0] // ECHELLE, img.size[1] // ECHELLE))
    img = img.convert("L")
    s = img.size[0] / W0
    gauche, haut = round((MARGE + dx) * s), round((Y0 + dy) * s)
    larg, haute = round((W0 - 2 * MARGE) * s), round((Y1 - Y0) * s)
    x = np.asarray(img.crop((gauche, haut, gauche + larg, haut + haute)).resize(TAILLE), dtype=np.float64)
    e = x.std()
    return ((x - x.mean()) / (e if e > 0 else 1.0)).astype(np.float32).ravel()


def main():
    cat = pd.read_csv(TRAITEMENT / "catalogue_images.csv", usecols=["fichier", "exploitable", "decalage_x", "decalage_y"])
    cat = cat.set_index("fichier")
    dec = cat[["decalage_x", "decalage_y"]].fillna(0.0)
    split = pd.read_csv(TRAITEMENT / "split_train_test.csv").set_index("fichier").split
    df = donnees()
    ann_train = df[df.split.isin(["train", "val"])].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    base = sorted(f for f in cat.index[(cat.exploitable == "oui") & cat.decalage_y.notna()] if split.get(f) == "train")
    print(f"POD : base {len(base)} photos train ; {len(ann_train)} annotées train+val ; {len(test)} test ; CPU {nom_cpu()}")

    # 1-2. lecture des photos train et POD
    t0 = time.perf_counter()
    with ThreadPoolExecutor(16) as ex:
        X = np.stack(list(ex.map(lambda f: charger(f, *dec.loc[f]), base)))
    xbar = X.mean(axis=0)
    A = X - xbar
    del X
    valeurs, vecteurs = np.linalg.eigh(A.astype(np.float64) @ A.T.astype(np.float64))
    ordre = np.argsort(valeurs)[::-1][:K_MAX]
    valeurs, vecteurs = np.clip(valeurs[ordre], 1e-12, None), vecteurs[:, ordre]
    modes = ((A.T @ vecteurs.astype(np.float32)) / np.sqrt(valeurs).astype(np.float32)).T     # (K_MAX, p)
    idx = {f: i for i, f in enumerate(base)}
    t_pod = time.perf_counter() - t0
    print(f"  POD : {len(base)} photos, {K_MAX} modes en {t_pod:.0f} s")

    # coefficients des photos annotées du train (celles hors de la base POD sont lues à part)
    def coefs(fichiers):
        lignes = []
        for f in fichiers:
            a = A[idx[f]] if f in idx else charger(f, *dec.loc[f]) - xbar
            lignes.append(a @ modes.T)
        return np.stack(lignes)

    t0 = time.perf_counter()
    C_train = coefs(ann_train.fichier)
    t_coefs = time.perf_counter() - t0
    h_train = (ann_train.niveau_px - ann_train.fichier.map(dec.decalage_y)).to_numpy()
    my, sy = h_train.mean(), h_train.std()

    # 3. ajustement des modèles
    ajustes = {}
    for cle, (nom, genre, K) in MODELES.items():
        t0 = time.perf_counter()
        sx = StandardScaler().fit(C_train[:, :K])
        Xtr = sx.transform(C_train[:, :K])
        if genre == "ridge":
            modeles = [Ridge(alpha=1.0).fit(Xtr, (h_train - my) / sy)]
            appris = "régression"
        else:
            modeles = [MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=3000,
                                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
                                    random_state=g).fit(Xtr, (h_train - my) / sy) for g in GRAINES]
            appris = f"~{np.mean([m.n_iter_ for m in modeles]):.0f} itér. ({len(GRAINES)} graines)"
        ajustes[cle] = (sx, modeles, K, appris, t_pod + t_coefs + time.perf_counter() - t0)

    # 4. test, une photo à la fois, un cœur : lecture + projection + prédiction
    lecture = Chrono()
    par_modele = {cle: Chrono() for cle in MODELES}
    preds = {cle: [] for cle in MODELES}
    with threadpool_limits(1):
        for f in test.fichier:
            dx, dy = dec.loc[f]
            with lecture:
                a = (charger(f, dx, dy) - xbar) @ modes.T
            for cle, (sx, modeles, K, _, _) in ajustes.items():
                with par_modele[cle]:
                    z = sx.transform(a[None, :K])
                    h = np.mean([m.predict(z)[0] for m in modeles]) * sy + my
                preds[cle].append(h + dy)
    for cle, (nom, _, _) in MODELES.items():
        sx, modeles, K, appris, duree = ajustes[cle]
        temps = 1000 * np.median(np.array(lecture.durees) + np.array(par_modele[cle].durees))
        enregistrer(cle, CATEGORIE, nom, test[["fichier", "niveau_px"]].assign(prediction_px=preds[cle]),
                    apprentissage=appris, duree_entrainement_s=duree, machine_entrainement=f"CPU {nom_cpu()}",
                    temps_image_ms=round(float(temps), 1), machine_image="CPU 1 cœur", K=K,
                    photos_base_pod=len(base), photos_annotees_train=len(ann_train))


if __name__ == "__main__":
    main()
