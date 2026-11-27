"""Diapos de reconstruction POD avec deux photos test côte à côte : niveau moyen (gauche) et eau haute (droite).
Utilise les figures de figures_pod.py (préfixe 4_) et exemple_eau_haute.py (préfixe 8_).

    .venv/bin/python presentation/creer_deux_exemples.py  ->  presentation/reconstruction_deux_exemples.pptx
"""
import math
from pathlib import Path
import numpy as np, pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

ICI = Path(__file__).resolve().parent; F = ICI / "figures"
BLEU, ROUGE, GRIS, NOIR = RGBColor(0x1D, 0x35, 0x57), RGBColor(0xE6, 0x39, 0x46), RGBColor(0x6C, 0x75, 0x7D), RGBColor(0x21, 0x25, 0x29)
KMAX = 3493
infos = dict(l.split("=") for l in (F / "infos.txt").read_text().split())
gauche = pd.read_csv(F / "4_reconstruction.csv").set_index("K")
# erreurs de l'exemple eau haute (calculées par exemple_eau_haute.py)
droite = {1: (23, 80), 4: (48, 58), 16: (68, 39), 64: (80, 32), 256: (86, 29)}
ng, nd = infos["photo_test"], "2026-06-24-151505.jpg"
date = lambda n: f"{n[8:10]}/{n[5:7]} à {n[11:13]}h{n[13:15]}"

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
def texte(s, x, y, l, hh, lignes, taille=16, couleur=NOIR, align=PP_ALIGN.LEFT):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(l), Inches(hh)).text_frame; tf.word_wrap = True
    for j, li in enumerate(lignes):
        t, opt = (li, {}) if isinstance(li, str) else li
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph(); p.alignment = align; p.space_after = Pt(opt.get("espace", 4))
        r = p.add_run(); r.text = t; r.font.size = Pt(opt.get("taille", taille)); r.font.bold = opt.get("gras", False)
        r.font.color.rgb = opt.get("couleur", couleur)
def nouvelle(titre, sous=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12)); b.fill.solid(); b.fill.fore_color.rgb = BLEU; b.line.fill.background()
    texte(s, 0.5, 0.3, 12.3, 0.8, [(titre, dict(taille=32, gras=True, couleur=BLEU))])
    if sous: texte(s, 0.5, 0.95, 12.3, 0.5, [sous], taille=16, couleur=GRIS)
    return s
img = lambda s, f, x, y, l=None, hh=None: s.shapes.add_picture(str(F / f), Inches(x), Inches(y), Inches(l) if l else None, Inches(hh) if hh else None)
COLS = [(0.45, "niveau moyen", ng), (6.85, "eau haute", nd)]; L = 6.05

# diapo d'introduction : les deux photos brutes
s = nouvelle("Reconstruire deux photos jamais vues", "Deux photos de test : elles n'ont pas servi à calculer les modes")
for (x, et, n), f in zip(COLS, ["4_photo_test.jpg", "8_photo_test_eau_haute.jpg"]):
    texte(s, x, 1.5, L, 0.4, [(f"{et} — {date(n)}", dict(taille=18, gras=True, couleur=BLEU))])
    img(s, f, x + (L - 5.4 * 1919 / 1753) / 2, 1.95, hh=5.0)

for k in (1, 4, 16, 64, 256):
    s = nouvelle(f"Reconstruction avec {k} mode{'s' if k > 1 else ''}")
    errs = [gauche.loc[k, "erreur"], droite[k][1]]
    for (x, et, n), pre, err in zip(COLS, ["4", "8"], errs):
        texte(s, x, 1.05, L, 0.35, [(f"{et} — {date(n)}", dict(taille=15, gras=True, couleur=BLEU))])
        texte(s, x, 1.4, L, 0.3, ["photo test (prétraitée)"], taille=12, couleur=GRIS)
        img(s, f"{pre}_test_pretraitee.png", x, 1.7, l=L)
        texte(s, x, 3.75, L, 0.3, [f"reconstruite avec {k} mode{'s' if k > 1 else ''}"], taille=12, couleur=GRIS)
        img(s, f"{pre}_reconstruction_{k:04d}.png", x, 4.05, l=L)
        texte(s, x, 6.08, L, 0.5, [(f"erreur de reconstruction : {err:.0f} %", dict(taille=18, gras=True, couleur=ROUGE))], align=PP_ALIGN.CENTER)
    texte(s, 0.45, 6.7, 6, 0.4, [(f"N = {k}  —  variance expliquée (photos d'entraînement) : {gauche.loc[k, 'variance']:.0f} %",
                                  dict(taille=14, gras=True, couleur=BLEU))])
    frac = math.log(k) / math.log(KMAX)
    fond = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.6), Inches(6.85), Inches(4.3), Inches(0.12)); fond.fill.solid()
    fond.fill.fore_color.rgb = RGBColor(0xDE, 0xE2, 0xE6); fond.line.fill.background()
    if frac > 0:
        b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.6), Inches(6.85), Inches(4.3 * frac), Inches(0.12)); b.fill.solid()
        b.fill.fore_color.rgb = BLEU; b.line.fill.background()
    texte(s, 8.6, 6.98, 4.3, 0.3, ["1 → 3 493 modes (échelle log)"], taille=10, couleur=GRIS)
prs.save(ICI / "reconstruction_deux_exemples.pptx"); print("écrit", len(prs.slides), "diapos")
