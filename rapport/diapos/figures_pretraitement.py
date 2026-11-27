"""Figures de la diapo « prétraitement » (découpe, réduction, noir et blanc, normalisation).

    .venv/bin/python presentation/figures_pretraitement.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir())
IMAGES, OUT = ROOT / "data" / "images", Path(__file__).resolve().parent / "figures"
W0, Y0, Y1, MARGE, ECHELLE = 1919, 500, 1100, 40, 4
TAILLE = ((W0 - 2 * MARGE) // ECHELLE, (Y1 - Y0) // ECHELLE)
cat = pd.read_csv(ROOT / "donnees" / "catalogue_images.csv").set_index("fichier")

def decouper(img, dx, dy):
    s = img.size[0] / W0
    g, t = round((MARGE + dx) * s), round((Y0 + dy) * s)
    return img.crop((g, t, g + round((W0 - 2 * MARGE) * s), t + round((Y1 - Y0) * s)))

def normalisee(gris):
    x = np.asarray(gris, dtype=np.float64); x = (x - x.mean()) / x.std()
    return Image.fromarray((np.clip(x / 6 + 0.5, 0, 1) * 255).astype(np.uint8))

AFFICHE = (920, 300)          # taille d'affichage commune (les images réduites sont agrandies sans lissage)
nom = (OUT / "infos.txt").read_text().split("photo_ref=")[1].split()[0]
bande = decouper(Image.open(IMAGES / nom), cat.at[nom, "decalage_x"], cat.at[nom, "decalage_y"])
petite = bande.resize(TAILLE, Image.BILINEAR)
gris = petite.convert("L")
bande.resize(AFFICHE).save(OUT / "p_1_bande.png")
petite.resize(AFFICHE, Image.NEAREST).save(OUT / "p_2_reduite.png")
gris.resize(AFFICHE, Image.NEAREST).save(OUT / "p_3_gris.png")
normalisee(gris).resize(AFFICHE, Image.NEAREST).save(OUT / "p_4_normalisee.png")

# effet de la normalisation : une photo sombre et une photo claire, même niveau d'eau environ
ann = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "donnees" / "annotations").glob("niveau_*.csv"))])
ann = ann.dropna(subset=["niveau_px"]).drop_duplicates("fichier").set_index("fichier").join(cat[["luminosite", "decalage_x", "decalage_y"]])
ann = ann[(ann["niveau_px"] - ann["decalage_y"]).between(835, 850) & pd.Series(ann.index.str[11:13].astype(int), index=ann.index).between(8, 17)]
# pas les extrêmes : une photo assez sombre (10 % les plus sombres) et une assez claire (10 % les plus claires)
lum = ann["luminosite"].sort_values()
sombre, claire = lum.index[int(0.10 * len(lum))], lum.index[int(0.90 * len(lum))]
for nom_f, et in ((sombre, "sombre"), (claire, "claire")):
    g = decouper(Image.open(IMAGES / nom_f), cat.at[nom_f, "decalage_x"], cat.at[nom_f, "decalage_y"]).resize(TAILLE).convert("L")
    g.resize(AFFICHE, Image.NEAREST).save(OUT / f"p_{et}_avant.png")
    normalisee(g).resize(AFFICHE, Image.NEAREST).save(OUT / f"p_{et}_apres.png")
open(OUT / "p_infos.txt", "w").write(f"sombre={sombre}\nclaire={claire}\n")
print(nom, sombre, claire)

# versions normalisées (noir et blanc, moyenne 0, écart type 1) des 8 bandes de la diapo « les photos se ressemblent »
for i in range(8):
    f = OUT / f"2_bande_{i}.png"
    if f.exists():
        normalisee(Image.open(f).resize(TAILLE).convert("L")).resize((920, 300), Image.NEAREST).save(OUT / f"2_bande_{i}_normalisee.png")
