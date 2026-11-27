"""Frise chronologique du découpage entraînement / test (photos exploitables par semaine).

    .venv/bin/python rapport/diapos/figures_split.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir())
OUT = Path(__file__).resolve().parent / "figures"
cat = pd.read_csv(ROOT / "donnees" / "catalogue_images.csv")
sp = pd.read_csv(ROOT / "donnees" / "split_train_test.csv")
d = cat.merge(sp, on="fichier")
d["date"] = pd.to_datetime(d.fichier.str[:10])
d["lundi"] = d.date - pd.to_timedelta(d.date.dt.day_of_week, unit="D")
ok = d[d.exploitable == "oui"]
sem = ok.groupby(["lundi", "split"]).size().unstack(fill_value=0)
BLEU, ROUGE, GRIS = "#1d3557", "#e63946", "#adb5bd"

plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(14, 4.6))
for lundi, r in sem.iterrows():
    for s, c in (("train", BLEU), ("test", ROUGE)):
        if r.get(s, 0):
            ax.bar(lundi + pd.Timedelta(days=3.5), r[s], width=6.2, color=c, zorder=2)
# trou sans photos
vide = (pd.Timestamp("2026-07-09"), pd.Timestamp("2026-08-21"))
ax.axvspan(*vide, color=GRIS, alpha=.25, zorder=0, lw=0)
ax.text(vide[0] + (vide[1] - vide[0]) / 2, sem.values.max() * 0.55, "pas de photos\n(9 juillet – 20 août)", ha="center", color="#495057", fontsize=13)
for lundi, r in sem.iterrows():
    if r.get("test", 0):
        ax.text(lundi + pd.Timedelta(days=3.5), sem.values.max() * 1.08, f"test\n{lundi:%d/%m}–{lundi + pd.Timedelta(days=6):%d/%m}",
                ha="center", va="bottom", color=ROUGE, fontsize=12, fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLEU, label=f"entraînement : {int(sem['train'].sum())} photos"),
                   Patch(color=ROUGE, label=f"test : {int(sem['test'].sum())} photos")], loc="upper center", bbox_to_anchor=(0.415, 1.0), frameon=False)
ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: ["janv.", "févr.", "mars", "avril", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."][mdates.num2date(v).month - 1]))
ax.set_ylabel("photos exploitables / semaine"); ax.set_ylim(0, sem.values.max() * 1.3)
plt.tight_layout(); fig.savefig(OUT / "0_decoupage.png", dpi=150)
print(sem.sum().to_dict())
