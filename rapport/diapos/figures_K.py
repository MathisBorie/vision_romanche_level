"""Figure de la diapo « choix du nombre de modes K ».

Valeurs : sorties du notebook 06 (POD noir et blanc à 1/4, 3 493 photos train annotées, 883 photos test,
réseau (64, 32) moyenné sur 3 graines, régression ridge).
    .venv/bin/python presentation/figures_K.py
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "figures"
K = [1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 3493]
train = [29.2, 19.5, 13.4, 9.0, 5.2, 4.1, 3.1, 2.0, 2.0, 1.7, 2.0, 1.8, 1.9, 2.0, 2.8, 7.4, 14.5]
test = [53.9, 27.4, 31.4, 25.4, 17.8, 14.3, 13.5, 14.7, 13.7, 11.7, 11.2, 10.6, 11.3, 11.7, 11.3, 31.5, 43.1]
lin = [46.9, 41.6, 38.1, 31.6, 23.0, 21.9, 17.1, 14.7, 15.1, 14.3, 13.6, 13.5, 13.5, 13.4, 13.2, 13.4, 13.0]
NAIF = 46.1
BLEU, ROUGE, VERT, GRIS = "#1d3557", "#e63946", "#2a9d8f", "#6c757d"

plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(11, 5.8))
ax.axvspan(1, 20, color=GRIS, alpha=.07); ax.axvspan(2500, 3600, color=GRIS, alpha=.07)
ax.plot(K, test, "o-", color=ROUGE, lw=3, ms=7, label="réseau : erreur sur le test")
ax.plot(K, train, "o--", color=ROUGE, lw=1.5, ms=4, alpha=.6, label="réseau : erreur sur l'entraînement")
ax.plot(K, lin, "s-", color=VERT, lw=2, ms=5, label="régression linéaire : erreur sur le test")
ax.axhline(NAIF, color=GRIS, ls=":", lw=2, label="prédire toujours la hauteur moyenne")
i = int(np.argmin(test))
ax.plot(K[i], test[i], "*", ms=22, color=ROUGE, mec="k", zorder=5)
ax.annotate(f"optimum : K = {K[i]}\n{str(test[i]).replace(".", ",")} px", (K[i], test[i]), (K[i] * 0.12, 4), fontsize=15, color=ROUGE,
            arrowprops=dict(arrowstyle="->", color=ROUGE))
ax.text(1.15, 58, "trop peu de modes :\nil manque de l'information", fontsize=12, color=GRIS, va="top")
ax.text(3450, 58, "trop de modes :\nle réseau apprend\nle bruit", fontsize=12, color=GRIS, va="top", ha="right")
ax.set_xscale("log"); ax.set_xlim(0.9, 3800); ax.set_ylim(0, 60)
ax.set_xlabel("K : nombre de coefficients POD donnés au réseau"); ax.set_ylabel("erreur moyenne sur la hauteur (px)")
ax.grid(alpha=.3, which="both"); ax.legend(loc="upper center", bbox_to_anchor=(0.52, 0.99), fontsize=12, framealpha=.95)
plt.tight_layout(); fig.savefig(OUT / "5_choix_K.png", dpi=150)
print("écrit", OUT / "5_choix_K.png")
