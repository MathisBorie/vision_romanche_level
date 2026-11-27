"""Construit presentation/partie_POD.pptx à partir des figures de presentation/figures/.

    .venv/bin/python presentation/figures_pod.py   # figures
    .venv/bin/python presentation/creer_pptx.py    # diapos
"""
from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ICI = Path(__file__).resolve().parent
F = ICI / "figures"
infos = dict(l.split("=") for l in (F / "infos.txt").read_text().split())
rec = pd.read_csv(F / "4_reconstruction.csv").set_index("K")
etiquettes = (F / "2_etiquettes.txt").read_text().splitlines()

BLEU, ROUGE, GRIS, NOIR = RGBColor(0x1D, 0x35, 0x57), RGBColor(0xE6, 0x39, 0x46), RGBColor(0x6C, 0x75, 0x7D), RGBColor(0x21, 0x25, 0x29)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
VIDE = prs.slide_layouts[6]


def texte(slide, x, y, l, h, lignes, taille=20, couleur=NOIR, gras=False, align=PP_ALIGN.LEFT):
    """lignes : liste de str ou de (str, dict(taille, couleur, gras))."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(l), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, ligne in enumerate(lignes):
        t, opt = (ligne, {}) if isinstance(ligne, str) else ligne
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(opt.get("espace", 8))
        r = p.add_run(); r.text = t
        r.font.size = Pt(opt.get("taille", taille)); r.font.bold = opt.get("gras", gras)
        r.font.color.rgb = opt.get("couleur", couleur)
    return tb


def nouvelle(titre, sous_titre=None):
    s = prs.slides.add_slide(VIDE)
    bande = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12))
    bande.fill.solid(); bande.fill.fore_color.rgb = BLEU; bande.line.fill.background()
    texte(s, 0.5, 0.3, 12.3, 0.8, [titre], taille=32, couleur=BLEU, gras=True)
    if sous_titre:
        texte(s, 0.5, 1.0, 12.3, 0.5, [sous_titre], taille=16, couleur=GRIS)
    return s


def image(slide, chemin, x, y, l=None, h=None):
    return slide.shapes.add_picture(str(chemin), Inches(x), Inches(y),
                                    Inches(l) if l else None, Inches(h) if h else None)


# 1. une photo = un vecteur énorme, et comment on le réduit
s = nouvelle("Une photo = un vecteur énorme : on le réduit avant la POD")


def fleche(slide, x, y, l, action, detail, larg_txt=None):
    """Flèche horizontale avec l'action au-dessus et le détail en dessous."""
    f = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(l), Inches(0.32))
    f.fill.solid(); f.fill.fore_color.rgb = BLEU; f.line.fill.background()
    lt = larg_txt or l + 0.4
    texte(slide, x + l / 2 - lt / 2, y - 0.62, lt, 0.6, [(action, dict(taille=13, gras=True, couleur=BLEU))], align=PP_ALIGN.CENTER)
    texte(slide, x + l / 2 - lt / 2, y + 0.36, lt, 0.9, [(detail, dict(taille=10, couleur=GRIS))], align=PP_ALIGN.CENTER)


def legende(slide, x, y, l, dims, n, rouge=False):
    texte(slide, x, y, l, 0.7, [(dims, dict(taille=12, couleur=GRIS, espace=0)),
                               (n, dict(taille=16, gras=True, couleur=ROUGE if rouge else NOIR))], align=PP_ALIGN.CENTER)


# ligne 1 : photo entière -> bande découpée
image(s, F / "1_photo_entiere.png", 0.5, 1.15, h=3.0)                     # ~3,3 in de large
legende(s, 0.5, 4.15, 3.3, "1 919 × 1 753 pixels × 3 couleurs", "≈ 10 millions de nombres")
fleche(s, 4.1, 2.45, 2.0, "Découper la rive", "on garde la bande autour de la rive (rectangle rouge), recalée sur le mouvement de la caméra", larg_txt=2.6)
image(s, F / "p_1_bande.png", 6.5, 1.75, l=5.6)
legende(s, 6.5, 3.65, 5.6, "1 839 × 600 pixels × 3 couleurs", "3,3 millions de nombres")

# ligne 2 : réduire -> noir et blanc -> normaliser
y2, L = 5.25, 2.7
etapes2 = [("Réduire 4 ×", "chaque bloc de 4 × 4 pixels → 1 pixel", "p_2_reduite.png", "459 × 150 × 3", "206 550 nombres", False),
           ("Noir et blanc", "3 couleurs → 1", "p_3_gris.png", "459 × 150", "68 850 nombres", True),
           ("Normaliser", "(pixel − moyenne) / écart type", "p_4_normalisee.png", "moyenne 0, écart type 1", "68 850 nombres", False)]
x = 0.3
for action, detail, f, dims, n, rouge in etapes2:
    fleche(s, x, y2 + 0.28, 1.1, action, detail, larg_txt=1.3)
    image(s, F / f, x + 1.3, y2, l=L)
    legende(s, x + 1.3, y2 + 0.92, L, dims, n, rouge=rouge)
    x += 1.3 + L + 0.15
texte(s, 0.5, 7.05, 12.3, 0.4, [("10 millions → 68 850 nombres (145 × moins). La normalisation ne réduit pas la taille : "
                                 "elle efface les différences de luminosité et de contraste entre photos.", dict(taille=13, couleur=GRIS))])

# 2. pourquoi normaliser
s = nouvelle("Pourquoi normaliser ?", "Deux photos au même niveau d'eau, l'une sombre, l'autre claire (noir et blanc, avant / après normalisation)")
pinfos = dict(l.split("=") for l in (F / "p_infos.txt").read_text().split())
texte(s, 2.3, 1.5, 5, 0.4, [("avant", dict(taille=18, gras=True, couleur=GRIS))], align=PP_ALIGN.CENTER)
texte(s, 7.7, 1.5, 5, 0.4, [("après normalisation", dict(taille=18, gras=True, couleur=BLEU))], align=PP_ALIGN.CENTER)
for j, (et, f) in enumerate([("sombre", pinfos["sombre"]), ("claire", pinfos["claire"])]):
    y = 2.0 + j * 1.95
    texte(s, 0.4, y + 0.45, 1.8, 0.8, [(f"photo {et}", dict(taille=16, gras=True, couleur=NOIR, espace=0)),
                                       (f"{f[8:10]}/{f[5:7]} à {f[11:13]}h{f[13:15]}", dict(taille=12, couleur=GRIS))])
    image(s, F / f"p_{et}_avant.png", 2.3, y, l=5.0)
    fl = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(7.38), Inches(y + 0.65), Inches(0.26), Inches(0.35))
    fl.fill.solid(); fl.fill.fore_color.rgb = BLEU; fl.line.fill.background()
    image(s, F / f"p_{et}_apres.png", 7.7, y, l=5.0)
texte(s, 0.5, 5.95, 12.3, 1.3, [
    ("Pour chaque photo : pixel  →  (pixel − moyenne de la photo) / écart type de la photo", dict(taille=17, gras=True, couleur=BLEU)),
    ("Toutes les photos ont alors une moyenne 0 et un écart type 1 : la POD compare la forme de la scène "
     "(le mur, les blocs, la rive), pas la luminosité ou le contraste de la journée.", dict(taille=15)),
])

# 3. les photos se ressemblent
s = nouvelle("Mais nos photos se ressemblent beaucoup",
             "8 photos d'entraînement prises au hasard, de mai à octobre, à toute heure")
for i, et in enumerate(etiquettes):
    col, lig = i // 4, i % 4
    x, y = 0.5 + col * 6.3, 1.55 + lig * 1.43
    texte(s, x, y - 0.05, 3, 0.3, [et], taille=12, couleur=GRIS)
    image(s, F / f"2_bande_{i}.png", x, y + 0.22, l=3.45)
texte(s, 4.15, 1.75, 2.1, 5.3, [
    ("Toujours pareil :", dict(taille=16, gras=True, couleur=BLEU)),
    ("le mur, les blocs, l'échelle, le cadrage", dict(taille=15, espace=18)),
], align=PP_ALIGN.LEFT)
texte(s, 10.45, 1.75, 2.6, 5.3, [
    ("Ce qui change :", dict(taille=16, gras=True, couleur=ROUGE)),
    ("le niveau d'eau", dict(taille=15)),
    ("la lumière, les ombres", dict(taille=15)),
    ("la texture de l'eau", dict(taille=15, espace=18)),
    ("→ l'information utile tient dans beaucoup moins que 68 850 nombres", dict(taille=15, gras=True)),
])

# 3 bis. découpage entraînement / test
s = nouvelle("Entraînement et test : des semaines entières, jamais mélangées",
             "Photos exploitables (de jour, caméra alignée) de la saison 2026, par semaine")
image(s, F / "0_decoupage.png", 0.45, 1.5, l=12.4)
texte(s, 0.5, 5.75, 6.1, 1.6, [
    ("Cycles de 5 semaines : 4 en entraînement, 1 en test", dict(taille=16, gras=True, couleur=BLEU, espace=4)),
    ("Une semaine entière va d'un seul côté : deux photos de la même journée (même niveau, même lumière) "
     "ne peuvent pas être l'une en entraînement et l'autre en test.", dict(taille=13)),
])
texte(s, 6.9, 5.75, 6.0, 1.6, [
    ("Ce qui sert à quoi", dict(taille=16, gras=True, couleur=BLEU, espace=4)),
    ("entraînement : calcul des modes POD et entraînement du réseau", dict(taille=13, espace=2)),
    ("test : jamais vu ni par la POD ni par le réseau, il sert seulement à mesurer l'erreur", dict(taille=13, espace=2)),
    ("3 semaines de test : fin juin (crue), début septembre, début octobre (celle de fin juillet tombe dans le trou sans photos)", dict(taille=12, couleur=GRIS)),
])

# 4. la base POD (une seule diapo : formule, variance expliquée, image moyenne et modes)
s = nouvelle("La base POD : les N directions qui expliquent le plus de variance")
texte(s, 0.45, 1.0, 12.4, 0.75, [
    ("photo  ≈  moyenne  +  a₁·mode₁  +  a₂·mode₂  +  …  +  aₙ·modeₙ", dict(taille=16, couleur=BLEU, gras=True, espace=0)),
    ("modes = vecteurs propres de la covariance des photos d'entraînement, rangés par variance décroissante ; "
     "une photo est résumée par ses N coefficients a₁ … aₙ", dict(taille=12, couleur=GRIS)),
])
en_modes = dict(l.split("=") for l in (F / "3_energie_modes.txt").read_text().split())
L, PAS_X, PAS_Y, X0, Y0_ = 3.0, 3.15, 1.3, 0.45, 1.85
image(s, F / "3_variance_expliquee.png", X0 + 0.9, Y0_ - 0.05, h=2 * PAS_Y - 0.1)       # 2 x 2 cases en haut à gauche
cases = [(r, c) for r in range(4) for c in range(4) if not (r < 2 and c < 2)]
vignettes = [("image moyenne", "3_moyenne.png")] + [(f"mode {k} : {float(en_modes[str(k)]):.2g} %", f"3_mode_{k}.png")
                                                    for k in list(range(1, 8)) + [50, 200, 800, 3200]]
for (r, c), (et, f) in zip(cases, vignettes):
    x, y = X0 + c * PAS_X, Y0_ + r * PAS_Y
    texte(s, x, y - 0.05, L, 0.3, [et], taille=12, couleur=GRIS)
    image(s, F / f, x, y + 0.22, l=L)
texte(s, X0, 7.08, 12.4, 0.35, [("rouge / bleu : plus clair / plus sombre que l'image moyenne ; % = part de la variance. "
                                 "Les premiers modes décrivent la rive, le mur, l'éclairage ; les derniers ne contiennent plus que la texture de l'eau.",
                                 dict(taille=11, couleur=GRIS))])

# 4. une photo test au hasard
s = nouvelle("Reconstruire une photo jamais vue",
             "Photo de test, niveau d’eau moyen : elle n’a pas servi à calculer les modes")
t = infos["photo_test"]
image(s, F / "4_photo_test.jpg", 0.5, 1.6, h=5.5)
texte(s, 6.9, 1.7, 6, 0.4, [f"{t[8:10]}/{t[5:7]}/{t[:4]} à {t[11:13]}h{t[13:15]}"], taille=16, couleur=GRIS)
image(s, F / "4_bande_test.png", 6.9, 2.2, l=6.0)
texte(s, 6.9, 4.4, 6.0, 2.5, [
    ("On la projette sur les modes : on calcule a₁, a₂, …", dict(taille=18)),
    ("puis on la reconstruit avec de plus en plus de modes.", dict(taille=18, espace=16)),
    ("Erreur de reconstruction = ‖photo − reconstruction‖ / ‖photo − image moyenne‖", dict(taille=14, couleur=GRIS)),
])

# 5. succession de reconstructions
KS = [1, 4, 16, 64, 256, 1024, int(infos["kmax"])]
for k in KS:
    r = rec.loc[k]
    s = nouvelle(f"Reconstruction avec {format(k, ',').replace(',', ' ')} mode{'s' if k > 1 else ''}{' (tous)' if k == int(infos['kmax']) else ''}")
    texte(s, 0.5, 1.15, 5, 0.35, ["photo test (prétraitée)"], taille=14, couleur=GRIS)
    image(s, F / "4_test_pretraitee.png", 0.5, 1.5, l=7.6)
    texte(s, 0.5, 4.05, 5, 0.35, [f"reconstruite avec {format(k, ',').replace(',', ' ')} mode{'s' if k > 1 else ''}"], taille=14, couleur=GRIS)
    image(s, F / f"4_reconstruction_{k:04d}.png", 0.5, 4.4, l=7.6)
    texte(s, 8.6, 1.5, 4.4, 5.5, [
        ("N", dict(taille=16, couleur=GRIS, espace=0)),
        (f"{k:,}".replace(",", " "), dict(taille=44, gras=True, couleur=BLEU, espace=20)),
        ("variance expliquée (photos d'entraînement)", dict(taille=14, couleur=GRIS, espace=0)),
        (f"{r['variance']:.0f} %", dict(taille=36, gras=True, couleur=BLEU, espace=20)),
        ("erreur de reconstruction de cette photo", dict(taille=14, couleur=GRIS, espace=0)),
        (f"{r['erreur']:.0f} %", dict(taille=36, gras=True, couleur=ROUGE)),
    ])
    # barre de progression (échelle log)
    import math
    frac = math.log(k) / math.log(int(infos["kmax"]))
    fond = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.6), Inches(6.75), Inches(4.2), Inches(0.12))
    fond.fill.solid(); fond.fill.fore_color.rgb = RGBColor(0xDE, 0xE2, 0xE6); fond.line.fill.background()
    if frac > 0:
        barre = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.6), Inches(6.75), Inches(4.2 * frac), Inches(0.12))
        barre.fill.solid(); barre.fill.fore_color.rgb = BLEU; barre.line.fill.background()
    texte(s, 8.6, 6.85, 4.2, 0.3, ["1 → 3 493 modes (échelle log)"], taille=10, couleur=GRIS)
    if k == int(infos["kmax"]):
        texte(s, 8.4, 4.9, 4.6, 1.6, [("Même avec tous les modes, la texture exacte de l'eau n'est pas reconstruite (elle n'existe dans "
                                       "aucune photo d'entraînement), mais le mur, les blocs et la rive le sont.", dict(taille=13, couleur=GRIS))])

# 6. choix du nombre de modes
s = nouvelle("Combien de modes donner au réseau ?",
             "Réseau de neurones : entrée = les K premiers coefficients POD de la photo, sortie = la hauteur d'eau")
image(s, F / "5_choix_K.png", 0.4, 1.5, l=8.6)
texte(s, 9.25, 1.6, 3.8, 5.6, [
    ("Trop peu de modes", dict(taille=17, gras=True, couleur=BLEU, espace=2)),
    ("le réseau ne voit pas assez la scène : grosse erreur partout (biais)", dict(taille=14, espace=14)),
    ("Trop de modes", dict(taille=17, gras=True, couleur=BLEU, espace=2)),
    ("les derniers modes ne contiennent que du bruit : le réseau l'apprend par cœur, l'erreur sur le test remonte (sur-apprentissage)",
     dict(taille=14, espace=14)),
    ("Entre 300 et 2 000 modes", dict(taille=17, gras=True, couleur=ROUGE, espace=2)),
    ("erreur ≈ 11 px, minimum 10,6 px pour K = 750", dict(taille=14, espace=14)),
    ("Erreur sur 883 photos test, jamais vues ni par la POD ni par le réseau.", dict(taille=12, couleur=GRIS)),
])

# 7. répartition de l'erreur
e = dict(l.split("=") for l in (F / "6_infos.txt").read_text().split())
s = nouvelle("Répartition de l'erreur : POD + réseau (K = 750)", "883 photos test")
image(s, F / "6_repartition_erreur.png", 0.4, 1.5, h=5.4)
for j, ligne in enumerate((F / "6_trois.txt").read_text().splitlines()):
    _, et, n, vraie, pr, er = ligne.split("|")
    y = 1.3 + j * 1.9
    texte(s, 8.3, y, 4.7, 0.35, [(f"{et} : erreur {float(er):+.0f} px".replace("+-", "−").replace("-", "−"), dict(taille=14, gras=True, couleur=BLEU))])
    image(s, F / f"6_trois_{j}.png", 8.3, y + 0.35, l=4.6)
texte(s, 8.3, 7.05, 4.7, 0.35, [("vert : annoté, rouge : prédit par le réseau", dict(taille=11, couleur=GRIS))])

prs.save(ICI / "partie_POD.pptx")
print("écrit :", ICI / "partie_POD.pptx", "|", len(prs.slides), "diapos")
