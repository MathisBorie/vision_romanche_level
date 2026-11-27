"""5 photos test où chaque modèle se trompe le plus (prédictions d'Aymen, branche data-classification)."""
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
from PIL import Image
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir())
D = ROOT / "apprentissage" / "comparaison" / "resultats"
MODELES = [("pod_reseau_750", "POD + réseau (K = 750)"), ("sam_zero_shot", "SAM sans entraînement"),
           ("sam_affine", "SAM décodeur affiné"), ("encodeur_dinov2", "DINOv2")]
COURT = {"pod_reseau_750": "POD", "sam_zero_shot": "SAM brut", "sam_affine": "SAM affiné", "encodeur_dinov2": "DINOv2"}
dfs = {k: pd.read_csv(D / f"{k}_predictions.csv").set_index("fichier") for k, _ in MODELES}
Y0, Y1 = 380, 1200
fig, ax = plt.subplots(len(MODELES), 5, figsize=(20, 3.9 * len(MODELES)))
lignes = []
for i, (k, nom) in enumerate(MODELES):
    df = dfs[k]
    pires = df.erreur_px.abs().sort_values(ascending=False).index[:5]
    for j, f in enumerate(pires):
        r = df.loc[f]
        im = Image.open(ROOT / "data/images" / f).convert("RGB")
        a = ax[i, j]
        a.imshow(im.crop((0, Y0, im.width, Y1)), extent=(0, im.width, Y1, Y0))
        a.axhline(r.niveau_px, color="lime", lw=2.2)
        a.axhline(r.prediction_px, color="red", lw=2.2, ls="--")
        autres = "  ".join(f"{COURT[kk]} {abs(dfs[kk].loc[f].erreur_px):.0f}" for kk, n in MODELES if kk != k)
        a.set_title(f"{f[:10]} {f[11:13]}h{f[13:15]} — erreur {abs(r.erreur_px):.0f} px", fontsize=11)
        a.set_xlabel(f"autres : {autres}", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        lignes.append((nom, f, r.niveau_px, round(r.prediction_px, 1), round(r.erreur_px, 1),
                       *[round(dfs[kk].loc[f].erreur_px, 1) for kk, _ in MODELES]))
    ax[i, 0].set_ylabel(nom, fontsize=14, fontweight="bold")
fig.suptitle("Les 5 pires photos test de chaque modèle — vert : vrai niveau, rouge pointillé : prédiction", fontsize=15)
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "figures" / "pires_5_par_modele.jpg", dpi=80)
cols = ["modele", "fichier", "niveau_px", "prediction_px", "erreur_px"] + [f"erreur_{k}" for k, _ in MODELES]
t = pd.DataFrame(lignes, columns=cols); t.to_csv(Path(__file__).resolve().parent / "pires_5_par_modele.csv", index=False)
pd.set_option("display.width", 250); print(t.to_string(index=False))
