"""Diapos d'introduction (contexte + sommaire), dans le style de la présentation de l'équipe.

    .venv/bin/python presentation/creer_intro.py
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ICI = Path(__file__).resolve().parent; F = ICI / "figures"
BARRE, NOIR, GRIS = RGBColor(0x1B, 0x5E, 0x84), RGBColor(0x21, 0x25, 0x29), RGBColor(0x6C, 0x75, 0x7D)
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

def diapo(titre, num):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.12), prs.slide_height)
    b.fill.solid(); b.fill.fore_color.rgb = BARRE; b.line.fill.background()
    t = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12), Inches(0.9)).text_frame
    r = t.paragraphs[0].add_run(); r.text = titre; r.font.size = Pt(36); r.font.color.rgb = NOIR
    n = s.shapes.add_textbox(Inches(12.4), Inches(6.95), Inches(0.6), Inches(0.4)).text_frame
    r = n.paragraphs[0].add_run(); r.text = str(num); r.font.size = Pt(12); r.font.color.rgb = GRIS
    return s

def legende(s, x, y, l, lignes):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(l), Inches(0.9)).text_frame; tf.word_wrap = True
    for i, (txt, taille, coul, gras) in enumerate(lignes):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = txt; r.font.size = Pt(taille); r.font.color.rgb = coul; r.font.bold = gras

# Contexte
s = diapo("Contexte", 3)
H = 4.6
s.shapes.add_picture(str(F / "barrage_livet.jpg"), Inches(0.6), Inches(1.5), height=Inches(H))          # 4:3 -> 6,13 in
s.shapes.add_picture(str(F / "camera_niveau_moyen_haut.jpg"), Inches(7.05), Inches(1.5), height=Inches(H))  # ~5,3 in
legende(s, 0.6, 6.15, 6.2, [("Barrage-prise d'eau de Livet sur la Romanche (EDF)", 18, NOIR, True),
                            ("Photo : Macquaria, CC BY-SA 4.0, Wikimedia Commons", 10, GRIS, False)])
legende(s, 7.05, 6.15, 5.6, [("Notre caméra : la berge, toutes les 15 min", 18, NOIR, True),
                             ("2 juin 2026, 12h30, photo brute", 10, GRIS, False)])

# Sommaire
s = diapo("Sommaire", 2)
tf = s.shapes.add_textbox(Inches(0.9), Inches(1.6), Inches(11), Inches(5)).text_frame
plan = [("Contexte", 0), ("Méthodes de résolution", 0), ("Traitement d'images classique", 1), ("SAM 2 (Segment Anything Model)", 1),
        ("POD (décomposition orthogonale propre)", 1), ("Encodeurs pré-entraînés", 1), ("Résultats", 0), ("Prochaines étapes", 0)]
for i, (txt, niv) in enumerate(plan):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.level = niv; p.space_after = Pt(10 if niv == 0 else 4)
    r = p.add_run(); r.text = ("•  " if niv == 0 else "–  ") + txt
    r.font.size = Pt(26 if niv == 0 else 20); r.font.color.rgb = NOIR if niv == 0 else GRIS
# l'ordre des diapos : sommaire puis contexte
ids = prs.slides._sldIdLst; ids.insert(0, ids[1])
prs.save(ICI / "introduction.pptx"); print("écrit", ICI / "introduction.pptx")
