"""Troisième exemple de reconstruction POD : une photo test en eau basse.
Mêmes diapos que l'exemple 1 (photo, puis reconstruction avec 1, 4, 16, 64, 256 modes).

    .venv/bin/python presentation/exemple_eau_haute.py   ->  presentation/reconstruction_eau_basse.pptx
"""
import glob, math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir()); ICI = Path(__file__).resolve().parent; F = ICI / "figures"
IMAGES = ROOT / "data" / "images"
W0, Y0, Y1, MARGE, E = 1919, 500, 1100, 40, 4
T = ((W0 - 2 * MARGE) // E, (Y1 - Y0) // E); h, w = T[1], T[0]
cat = pd.read_csv(ROOT / "donnees" / "catalogue_images.csv").set_index("fichier")
split = pd.read_csv(ROOT / "donnees" / "split_train_test.csv").set_index("fichier")["split"]
ann = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "donnees" / "annotations" / "niveau_*.csv")))])
ann = ann.dropna(subset=["niveau_px"]).drop_duplicates("fichier").set_index("fichier")
haut = ann["niveau_px"] - cat["decalage_y"].reindex(ann.index)

def decouper(img, dx, dy):
    s = img.size[0] / W0; g, t = round((MARGE + dx) * s), round((Y0 + dy) * s)
    return img.crop((g, t, g + round((W0 - 2 * MARGE) * s), t + round((Y1 - Y0) * s)))
def charger(nom):
    img = Image.open(IMAGES / nom); img.draft("L", (img.size[0] // E, img.size[1] // E))
    x = np.asarray(decouper(img.convert("L"), cat.at[nom, "decalage_x"], cat.at[nom, "decalage_y"]).resize(T), dtype=np.float64)
    return ((x - x.mean()) / x.std()).astype(np.float32).ravel()
def en_image(v):
    rgba = plt.get_cmap("gray")(np.clip(v.reshape(h, w) / 6 + 0.5, 0, 1))
    return Image.fromarray((rgba[..., :3] * 255).astype(np.uint8)).resize((w * 2, h * 2), Image.BICUBIC)

utiles = cat.index[(cat.exploitable == "oui") & cat.decalage_y.notna()]
train = sorted(f for f in utiles if split.get(f) == "train"); test = sorted(f for f in utiles if split.get(f) == "test")
with ThreadPoolExecutor(8) as ex:
    X = np.stack(list(ex.map(charger, train)))
xbar = X.mean(0); A = X - xbar
val, vec = np.linalg.eigh((A @ A.T).astype(np.float64)); o = np.argsort(val)[::-1]
val, vec = np.clip(val[o], 0, None)[:-1], vec[:, o][:, :-1]; sig = np.sqrt(val); energie = val / val.sum(); KMAX = len(val)

# photo test en eau basse (hauteur > 850 px), de jour, erreur de reconstruction finale « raisonnable »
cand = [f for f in test if haut.get(f, -np.inf) > 850 and 12 <= int(f[11:13]) <= 16 and cat.at[f, "visibilite"] == "nette"
        and cat.at[f, "reflets"] == 0 and f[:7] == "2026-10"]          # octobre, sans reflets ni taches de soleil
with ThreadPoolExecutor(8) as ex:
    Xc = np.stack(list(ex.map(charger, cand)))
Dc = (Xc - xbar).astype(np.float64); ac = (Dc @ A.T.astype(np.float64)) @ vec / sig
e_tous = np.sqrt(np.clip((Dc ** 2).sum(1) - (ac ** 2).sum(1), 0, None) / (Dc ** 2).sum(1))
print(len(cand), "photos test eau basse ; erreur finale :", np.round(np.sort(e_tous)[:10], 2))
i = int(np.argmin(np.abs(e_tous - 0.235))); nom = cand[i]; x = Xc[i]; a = ac[i]; nx = np.sqrt((Dc[i] ** 2).sum())
print("photo choisie :", nom, "hauteur", round(haut[nom]), "erreur finale", round(100 * e_tous[i]))

Image.open(IMAGES / nom).save(F / "9_photo_test_eau_basse.jpg", quality=90)
decouper(Image.open(IMAGES / nom), cat.at[nom, "decalage_x"], cat.at[nom, "decalage_y"]).save(F / "9_bande_test_eau_basse.png")
en_image(x).save(F / "9_test_pretraitee.png")
KS = [1, 4, 16, 64, 256]; lignes = {}
for k in KS:
    rec = xbar + A.T @ (vec[:, :k] @ (a[:k] / sig[:k])).astype(np.float32)
    en_image(rec).save(F / f"9_reconstruction_{k:04d}.png")
    lignes[k] = (100 * energie[:k].sum(), 100 * np.linalg.norm(x - rec) / nx)

# diapos, même mise en page que l'exemple 1
BLEU, ROUGE, GRIS, NOIR = RGBColor(0x1D, 0x35, 0x57), RGBColor(0xE6, 0x39, 0x46), RGBColor(0x6C, 0x75, 0x7D), RGBColor(0x21, 0x25, 0x29)
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
def texte(s, x, y, l, hh, lignes_t, taille=20, couleur=NOIR):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(l), Inches(hh)).text_frame; tf.word_wrap = True
    for j, li in enumerate(lignes_t):
        t, opt = (li, {}) if isinstance(li, str) else li
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph(); p.space_after = Pt(opt.get("espace", 8))
        r = p.add_run(); r.text = t; r.font.size = Pt(opt.get("taille", taille)); r.font.bold = opt.get("gras", False)
        r.font.color.rgb = opt.get("couleur", couleur)
def nouvelle(titre, sous=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12)); b.fill.solid(); b.fill.fore_color.rgb = BLEU; b.line.fill.background()
    texte(s, 0.5, 0.3, 12.3, 0.8, [(titre, dict(taille=32, gras=True, couleur=BLEU))])
    if sous: texte(s, 0.5, 1.0, 12.3, 0.5, [sous], taille=16, couleur=GRIS)
    return s
img = lambda s, f, x, y, l=None, hh=None: s.shapes.add_picture(str(F / f), Inches(x), Inches(y), Inches(l) if l else None, Inches(hh) if hh else None)

s = nouvelle("Troisième exemple : eau basse", "Photo de test : elle n'a pas servi à calculer les modes")
img(s, "9_photo_test_eau_basse.jpg", 0.5, 1.6, hh=5.5)
texte(s, 6.9, 1.7, 6, 0.4, [f"{nom[8:10]}/{nom[5:7]}/{nom[:4]} à {nom[11:13]}h{nom[13:15]}"], taille=16, couleur=GRIS)
img(s, "9_bande_test_eau_basse.png", 6.9, 2.2, l=6.0)
for k in KS:
    var, err = lignes[k]
    s = nouvelle(f"Reconstruction avec {k} mode{'s' if k > 1 else ''}")
    texte(s, 0.5, 1.15, 5, 0.35, ["photo test (prétraitée)"], taille=14, couleur=GRIS); img(s, "9_test_pretraitee.png", 0.5, 1.5, l=7.6)
    texte(s, 0.5, 4.05, 5, 0.35, [f"reconstruite avec {k} mode{'s' if k > 1 else ''}"], taille=14, couleur=GRIS)
    img(s, f"9_reconstruction_{k:04d}.png", 0.5, 4.4, l=7.6)
    texte(s, 8.6, 1.5, 4.4, 5.5, [("N", dict(taille=16, couleur=GRIS, espace=0)), (f"{k}", dict(taille=44, gras=True, couleur=BLEU, espace=20)),
                                  ("variance expliquée (photos d'entraînement)", dict(taille=14, couleur=GRIS, espace=0)),
                                  (f"{var:.0f} %", dict(taille=36, gras=True, couleur=BLEU, espace=20)),
                                  ("erreur de reconstruction de cette photo", dict(taille=14, couleur=GRIS, espace=0)),
                                  (f"{err:.0f} %", dict(taille=36, gras=True, couleur=ROUGE))])
    frac = math.log(k) / math.log(KMAX)
    fond = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.6), Inches(6.75), Inches(4.2), Inches(0.12)); fond.fill.solid()
    fond.fill.fore_color.rgb = RGBColor(0xDE, 0xE2, 0xE6); fond.line.fill.background()
    if frac > 0:
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.6), Inches(6.75), Inches(4.2 * frac), Inches(0.12)); bar.fill.solid()
        bar.fill.fore_color.rgb = BLEU; bar.line.fill.background()
    texte(s, 8.6, 6.85, 4.2, 0.3, ["1 → 3 493 modes (échelle log)"], taille=10, couleur=GRIS)
prs.save(ICI / "reconstruction_eau_basse.pptx")
print({k: (round(v[0]), round(v[1])) for k, v in lignes.items()})
