"""Annotation manuelle du niveau d'eau (niveau_px) dans catalogue_images.csv.

Une fenêtre affiche chaque image ; un clic place une LIGNE ROUGE horizontale sur la frontière
eau / berge d'en face ; on l'ajuste au clavier puis on valide. Chaque validation est enregistrée
tout de suite dans le CSV : on peut fermer et reprendre plus tard, le script repart à la
première image non annotée.

Utilisation :
    .venv/bin/python data_treatment/define_water_lvl.py                    # images oui + douteux, ordre chronologique
    .venv/bin/python data_treatment/define_water_lvl.py --jour 2026-09     # seulement septembre (préfixe de date)
    .venv/bin/python data_treatment/define_water_lvl.py --aleatoire        # ordre aléatoire
    .venv/bin/python data_treatment/define_water_lvl.py --revoir           # inclut les images déjà annotées

Commandes :
    clic gauche          place la ligne rouge (recliquer pour la déplacer)
    flèche haut / bas    déplace la ligne de 1 px   (Maj + flèche : 10 px)
    p                    place la ligne sur la suggestion (pointillés jaunes, voir plus bas)
    Entrée ou espace     valide : niveau_px = y de la ligne, exploitable = oui, verifie = 1
    x                    rejette l'image : exploitable = non, verifie = 1, niveau_px vidé
    flèche droite        passe sans rien enregistrer
    flèche gauche        revient à l'image précédente
    q                    quitte (tout est déjà enregistré)
    Loupe / main de la barre d'outils : zoom / déplacement (les clics ne placent pas la ligne
    tant qu'un de ces outils est actif ; maison = vue d'origine).

Convention : niveau_px est le y (px, image d'origine 1919 x 1753) de la frontière eau / berge
d'en face, en moyenne sur la largeur de l'image (comme dans le notebook 04). Plus y est petit,
plus l'eau est haute. La caméra bouge : pour comparer deux images, utiliser niveau_px - decalage_y.

Suggestion (pointillés jaunes) : le niveau de la dernière image annotée dans les 3 jours
précédents, corrigé du mouvement de la caméra. Elle est là pour aller vite quand le niveau
n'a pas changé : vérifier qu'elle tombe bien sur la frontière avant de l'accepter.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent   # racine du projet (data/)
CSV = Path(__file__).resolve().parent / "catalogue_images.csv"   # à côté des scripts
IMAGES = ROOT / "data" / "images"
VUE_Y = (300, 1350)            # vue de départ : zone où se trouve la frontière
SUGGESTION_MAX = pd.Timedelta(days=3)

TOUCHES = {"enter", "return", " ", "x", "p", "q", "left", "right", "up", "down", "shift+up", "shift+down"}


def liberer_touches():
    """Retire nos touches des raccourcis par défaut de matplotlib (p = pan, q = quitter...)."""
    for k, v in plt.rcParams.items():
        if k.startswith("keymap.") and isinstance(v, list):
            plt.rcParams[k] = [t for t in v if t not in TOUCHES]


def ecrire_csv(df: pd.DataFrame, chemin: Path):
    tmp = chemin.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, chemin)    # écriture atomique : pas de CSV à moitié écrit si on coupe


class Annotateur:
    def __init__(self, csv: Path, images: Path, jour: str = "", aleatoire: bool = False,
                 revoir: bool = False):
        self.csv, self.images = csv, images
        self.df = pd.read_csv(csv, dtype=str, keep_default_na=False)
        self.t = pd.to_datetime(self.df.datetime)
        a_faire = self.df.exploitable != "non"
        if not revoir:
            a_faire &= self.df.niveau_px == ""
        if jour:
            a_faire &= self.df.jour.str.startswith(jour)
        self.liste = list(self.df.index[a_faire].sort_values(key=lambda i: self.t[i]))
        if aleatoire:
            np.random.default_rng().shuffle(self.liste)
        self.i = 0
        self.y = None              # position de la ligne rouge
        self.fig = self.ax = None

    # ---------- données ----------
    def ligne(self):
        return self.df.loc[self.liste[self.i]]

    @staticmethod
    def _float(v):
        try:
            return float(v)
        except ValueError:
            return None

    def suggestion(self):
        """Dernier niveau annoté dans les 3 jours avant l'image courante, corrigé du décalage."""
        idx = self.liste[self.i]
        t0 = self.t[idx]
        ok = (self.df.niveau_px != "") & (self.t < t0) & (self.t >= t0 - SUGGESTION_MAX)
        if not ok.any():
            return None
        j = self.t[ok].idxmax()
        y = float(self.df.at[j, "niveau_px"])
        dy_avant, dy = self._float(self.df.at[j, "decalage_y"]), self._float(self.df.at[idx, "decalage_y"])
        if dy_avant is not None and dy is not None:
            y += dy - dy_avant
        return y

    def enregistrer(self, **valeurs):
        idx = self.liste[self.i]
        for c, v in valeurs.items():
            self.df.at[idx, c] = v
        self.df.at[idx, "verifie"] = "1"
        ecrire_csv(self.df, self.csv)

    # ---------- actions ----------
    def valider(self):
        if self.y is None:
            self.message("Cliquer d'abord sur la frontière eau / berge", "red")
            return
        self.enregistrer(niveau_px=str(int(round(self.y))), exploitable="oui")
        self.suivante()

    def rejeter(self):
        self.enregistrer(niveau_px="", exploitable="non")
        self.suivante()

    def suivante(self):
        self.aller(self.i + 1)

    def precedente(self):
        self.aller(self.i - 1)

    def aller(self, i):
        if i < 0:
            return
        if i >= len(self.liste):
            self.message("Terminé : plus d'image à annoter. Fermer la fenêtre (q).", "green")
            return
        self.i = i
        self.afficher()

    def placer(self, y):
        self.y = float(np.clip(y, 0, self.hauteur - 1))
        self.trait.set_ydata([self.y, self.y])
        self.trait.set_visible(True)
        self.maj_titre()
        self.fig.canvas.draw_idle()

    # ---------- affichage ----------
    def demarrer(self):
        if not self.liste:
            print("Aucune image à annoter avec ces options.")
            return
        liberer_touches()
        self.fig, self.ax = plt.subplots(figsize=(15, 9))
        self.fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.08)
        self.fig.text(0.01, 0.015,
                      "clic : ligne rouge   |   haut/bas : 1 px (Maj : 10 px)   |   p : suggestion   |   "
                      "Entrée/espace : valider   |   x : rejeter   |   droite : passer   |   "
                      "gauche : précédente   |   q : quitter", fontsize=9)
        self.info = self.fig.text(0.99, 0.015, "", fontsize=10, ha="right", weight="bold")
        self.fig.canvas.mpl_connect("button_press_event", self.on_clic)
        self.fig.canvas.mpl_connect("key_press_event", self.on_touche)
        self.afficher()
        plt.show()

    def afficher(self):
        r = self.ligne()
        im = np.asarray(Image.open(self.images / r.fichier).convert("RGB"))
        self.hauteur, largeur = im.shape[:2]
        self.ax.clear()
        self.ax.imshow(im)
        self.ax.set_xlim(0, largeur)
        self.ax.set_ylim(*VUE_Y[::-1])
        self.trait = self.ax.axhline(0, color="red", lw=1.5, visible=False)
        self.sugg = self.suggestion()
        if self.sugg is not None:
            self.ax.axhline(self.sugg, color="yellow", lw=1, ls="--")
        self.y = None
        if r.niveau_px:                       # image déjà annotée (--revoir ou retour arrière)
            self.placer(float(r.niveau_px))
        self.maj_titre()
        self.message("")
        self.fig.canvas.draw_idle()

    def maj_titre(self):
        r = self.ligne()
        etat = [r.periode, r.visibilite or "-", "reflets" if r.reflets == "1" else "",
                f"exploitable={r.exploitable}", f"decalage_y={r.decalage_y or '?'}"]
        y = f"ligne y = {self.y:.0f}" if self.y is not None else "ligne : cliquer"
        s = f"   suggestion y = {self.sugg:.0f}" if self.sugg is not None else ""
        n_faits = (self.df.loc[self.liste, "niveau_px"] != "").sum()
        self.ax.set_title(f"{r.fichier}  ({r.heure} UTC)   [{self.i + 1}/{len(self.liste)}, {n_faits} annotées]\n"
                          f"{'  '.join(e for e in etat if e)}      {y}{s}", loc="left", fontsize=11)

    def message(self, texte, couleur="black"):
        self.info.set_text(texte)
        self.info.set_color(couleur)
        self.fig.canvas.draw_idle()

    # ---------- événements ----------
    def on_clic(self, ev):
        barre = getattr(self.fig.canvas, "toolbar", None)
        if ev.inaxes is not self.ax or ev.button != 1 or (barre is not None and barre.mode):
            return
        self.placer(ev.ydata)

    def on_touche(self, ev):
        k = ev.key
        if k in ("enter", "return", " "):
            self.valider()
        elif k == "x":
            self.rejeter()
        elif k == "p":
            if self.sugg is not None:
                self.placer(self.sugg)
        elif k == "right":
            self.suivante()
        elif k == "left":
            self.precedente()
        elif k == "q":
            plt.close(self.fig)
        elif k in ("up", "down", "shift+up", "shift+down") and self.y is not None:
            pas = 10 if k.startswith("shift") else 1
            self.placer(self.y + (-pas if k.endswith("up") else pas))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jour", default="", help="préfixe de date, ex. 2026-09 ou 2026-09-15")
    p.add_argument("--aleatoire", action="store_true", help="ordre aléatoire au lieu de chronologique")
    p.add_argument("--revoir", action="store_true", help="inclure les images déjà annotées")
    p.add_argument("--csv", type=Path, default=CSV)
    p.add_argument("--images", type=Path, default=IMAGES)
    a = p.parse_args()
    ann = Annotateur(a.csv, a.images, a.jour, a.aleatoire, a.revoir)
    print(f"{len(ann.liste)} images à annoter. Enregistrement dans {a.csv}")
    ann.demarrer()
    n = (ann.df.niveau_px != "").sum()
    print(f"{n} images annotées au total dans le CSV.")


if __name__ == "__main__":
    main()
