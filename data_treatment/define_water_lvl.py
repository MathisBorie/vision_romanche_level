"""Annotation manuelle du niveau d'eau (niveau_px), à trois sans annoter deux fois la même image.

Les jours sont répartis entre les annotateurs (jour de l'année modulo 3) : chacun a son ensemble,
réparti sur toute la saison. Chacun enregistre dans SON fichier :
    data_treatment/annotations/niveau_<nom>.csv     (fichier, niveau_px, exploitable, annote_le)
Un fichier par personne : pas de conflit git. classification.py reporte ensuite toutes les
annotations dans catalogue_images.csv (niveau_px, exploitable, verifie = 1, annotateur).

Utilisation :
    .venv/bin/python data_treatment/define_water_lvl.py aymen                  # son ensemble, ordre chronologique
    .venv/bin/python data_treatment/define_water_lvl.py mathis --jour 2026-09  # seulement septembre
    .venv/bin/python data_treatment/define_water_lvl.py tristan --revoir       # inclut ses images déjà annotées
    (--aleatoire : ordre aléatoire)

Commandes :
    clic gauche          ENREGISTRE le niveau à la hauteur du clic et passe à l'image suivante
    Entrée ou p          enregistre la suggestion (pointillés jaunes) et passe à la suivante
    x                    rejette l'image (exploitable = non) et passe à la suivante
    flèche droite        passe sans rien enregistrer
    flèche gauche        revient à l'image précédente (ligne rouge = niveau enregistré ;
                         recliquer pour le corriger)
    q                    quitte (tout est déjà enregistré ; au prochain lancement, reprise
                         à la première image non annotée)
    Loupe / main de la barre d'outils : zoom / déplacement (les clics n'enregistrent rien tant
    qu'un de ces outils est actif ; maison = vue d'origine).

Convention : niveau_px est le y (px, image d'origine 1919 x 1753) de la frontière eau / berge
d'en face, en moyenne sur la largeur de l'image (comme dans le notebook 04). Plus y est petit,
plus l'eau est haute. La caméra bouge : pour comparer deux images, utiliser niveau_px - decalage_y.

Suggestion (pointillés jaunes) : le niveau de la dernière image annotée (par n'importe qui) dans
les 4 jours précédents, corrigé du mouvement de la caméra. Vérifier qu'elle tombe bien sur la
frontière avant de l'accepter.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ICI = Path(__file__).resolve().parent
ROOT = ICI.parent                                  # racine du projet (data/)
CSV = ICI / "catalogue_images.csv"
ANNOTATIONS = ICI / "annotations"
IMAGES = ROOT / "data" / "images"

ANNOTATEURS = ["aymen", "mathis", "tristan"]     # ensemble d'un jour = ANNOTATEURS[jour de l'année % 3]
COLONNES = ["fichier", "niveau_px", "exploitable", "annote_le"]
VUE_Y = (300, 1350)                                # vue de départ : zone où se trouve la frontière
SUGGESTION_MAX = pd.Timedelta(days=4)

TOUCHES = {"enter", "return", "x", "p", "q", "left", "right"}


def annotateur_du_jour(jour: str) -> str:
    return ANNOTATEURS[pd.Timestamp(jour).day_of_year % len(ANNOTATEURS)]


def fichier_annotations(nom: str, dossier: Path = ANNOTATIONS) -> Path:
    return dossier / f"niveau_{nom}.csv"


def lire_annotations(dossier: Path = ANNOTATIONS) -> pd.DataFrame:
    """Toutes les annotations (tous les fichiers niveau_*.csv), avec la colonne annotateur."""
    parts = []
    for f in sorted(dossier.glob("niveau_*.csv")):
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        d["annotateur"] = f.stem.removeprefix("niveau_")
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=COLONNES + ["annotateur"])
    return pd.concat(parts, ignore_index=True)


def liberer_touches():
    """Retire nos touches des raccourcis par défaut de matplotlib (p = pan, q = quitter...)."""
    for k, v in plt.rcParams.items():
        if k.startswith("keymap.") and isinstance(v, list):
            plt.rcParams[k] = [t for t in v if t not in TOUCHES]


class Annotateur:
    def __init__(self, nom: str, csv: Path = CSV, images: Path = IMAGES, annotations: Path = ANNOTATIONS,
                 jour: str = "", aleatoire: bool = False, revoir: bool = False):
        self.nom, self.images = nom, images
        self.df = pd.read_csv(csv, dtype=str, keep_default_na=False).set_index("fichier")
        self.t = pd.to_datetime(self.df.datetime)

        self.sortie = fichier_annotations(nom, annotations)
        toutes = lire_annotations(annotations)
        self.mes = toutes[toutes.annotateur == nom].set_index("fichier")[COLONNES[1:]]
        deja = set(toutes.fichier) | set(self.df.index[self.df.verifie == "1"])

        # niveaux connus (pour la suggestion) : catalogue + tous les fichiers d'annotation
        self.niveaux = {f: float(v) for f, v in self.df.niveau_px.items() if v}
        self.niveaux.update({r.fichier: float(r.niveau_px) for r in toutes.itertuples() if r.niveau_px})
        for f in toutes.fichier[toutes.niveau_px == ""]:
            self.niveaux.pop(f, None)

        a_faire = (self.df.jour.map(annotateur_du_jour) == nom) & (self.df.exploitable != "non")
        if jour:
            a_faire &= self.df.jour.str.startswith(jour)
        pas_faites = ~self.df.index.isin(list(deja))
        if revoir:
            pas_faites |= self.df.index.isin(list(self.mes.index))
        self.liste = sorted(self.df.index[a_faire & pas_faites], key=lambda f: self.t[f])
        if aleatoire:
            np.random.default_rng().shuffle(self.liste)
        self.i = 0
        self.fig = self.ax = None

    # ---------- données ----------
    def courant(self) -> str:
        return self.liste[self.i]

    def decalage(self, f):
        v = self.df.at[f, "decalage_y"]
        return float(v) if v else None

    def suggestion(self):
        """Dernier niveau annoté dans les 4 jours avant l'image courante, corrigé du décalage."""
        f0 = self.courant()
        t0 = self.t[f0]
        avant = [f for f in self.niveaux if t0 - SUGGESTION_MAX <= self.t[f] < t0]
        if not avant:
            return None
        f = max(avant, key=lambda x: self.t[x])
        y = self.niveaux[f]
        d_avant, d = self.decalage(f), self.decalage(f0)
        if d_avant is not None and d is not None:
            y += d - d_avant
        return y

    def enregistrer(self, niveau: str, exploitable: str):
        f = self.courant()
        self.mes.loc[f] = [niveau, exploitable, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")]
        if niveau:
            self.niveaux[f] = float(niveau)
        else:
            self.niveaux.pop(f, None)
        self.sortie.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.sortie.with_suffix(".tmp")
        self.mes.sort_index().rename_axis("fichier").reset_index().to_csv(tmp, index=False)
        os.replace(tmp, self.sortie)    # écriture atomique : pas de fichier à moitié écrit si on coupe

    # ---------- actions ----------
    def valider(self, y: float, source: str = ""):
        y = int(round(float(np.clip(y, 0, self.hauteur - 1))))
        f = self.courant()
        self.enregistrer(str(y), "oui")
        self.suivante(f"{f[:-4]} : niveau {y} enregistré{source}")

    def rejeter(self):
        f = self.courant()
        self.enregistrer("", "non")
        self.suivante(f"{f[:-4]} : rejetée")

    def suivante(self, msg=""):
        self.aller(self.i + 1, msg)

    def aller(self, i, msg=""):
        if i < 0:
            return
        if i >= len(self.liste):
            self.message(f"{msg}   —   Terminé : plus d'image à annoter (q pour quitter).", "green")
            return
        self.i = i
        self.afficher(msg)

    # ---------- affichage ----------
    def demarrer(self):
        if not self.liste:
            print("Aucune image à annoter avec ces options.")
            return
        liberer_touches()
        self.fig, self.ax = plt.subplots(figsize=(15, 9))
        self.fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.08)
        self.fig.text(0.01, 0.015,
                      "clic : enregistre le niveau   |   Entrée / p : accepte la suggestion   |   x : rejeter   |   "
                      "droite : passer   |   gauche : précédente   |   q : quitter", fontsize=9)
        self.info = self.fig.text(0.99, 0.015, "", fontsize=10, ha="right", weight="bold")
        self.fig.canvas.mpl_connect("button_press_event", self.on_clic)
        self.fig.canvas.mpl_connect("key_press_event", self.on_touche)
        self.afficher()
        plt.show()

    def afficher(self, msg=""):
        f = self.courant()
        im = np.asarray(Image.open(self.images / f).convert("RGB"))
        self.hauteur, largeur = im.shape[:2]
        self.ax.clear()
        self.ax.imshow(im)
        self.ax.set_xlim(0, largeur)
        self.ax.set_ylim(*VUE_Y[::-1])
        self.sugg = self.suggestion()
        if self.sugg is not None:
            self.ax.axhline(self.sugg, color="yellow", lw=1, ls="--")
        if f in self.mes.index and self.mes.at[f, "niveau_px"]:   # déjà annotée (retour arrière, --revoir)
            self.ax.axhline(float(self.mes.at[f, "niveau_px"]), color="red", lw=1.5)

        r = self.df.loc[f]
        etat = [r.periode, r.visibilite or "-", "reflets" if r.reflets == "1" else "",
                f"exploitable={r.exploitable}", f"decalage_y={r.decalage_y or '?'}"]
        if f in self.mes.index:
            etat.append(f"DÉJÀ ENREGISTRÉE : {self.mes.at[f, 'niveau_px'] or 'rejetée'}")
        s = f"suggestion y = {self.sugg:.0f}" if self.sugg is not None else "pas de suggestion"
        n_faits = self.mes.index.isin(self.liste).sum()
        self.ax.set_title(f"{self.nom} — {f}  ({r.heure} UTC)   [{self.i + 1}/{len(self.liste)}, "
                          f"{n_faits} faites]\n{'  '.join(e for e in etat if e)}      {s}",
                          loc="left", fontsize=11)
        self.message(msg, "darkgreen")

    def message(self, texte, couleur="black"):
        self.info.set_text(texte)
        self.info.set_color(couleur)
        self.fig.canvas.draw_idle()

    # ---------- événements ----------
    def on_clic(self, ev):
        barre = getattr(self.fig.canvas, "toolbar", None)
        if ev.inaxes is not self.ax or ev.button != 1 or (barre is not None and barre.mode):
            return
        self.valider(ev.ydata)

    def on_touche(self, ev):
        k = ev.key
        if k in ("enter", "return", "p"):
            if self.sugg is None:
                self.message("Pas de suggestion pour cette image : cliquer sur la frontière", "red")
            else:
                self.valider(self.sugg, " (suggestion)")
        elif k == "x":
            self.rejeter()
        elif k == "right":
            self.suivante()
        elif k == "left":
            self.aller(self.i - 1)
        elif k == "q":
            plt.close(self.fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("nom", choices=ANNOTATEURS, help="qui annote (définit l'ensemble d'images)")
    p.add_argument("--jour", default="", help="préfixe de date, ex. 2026-09 ou 2026-09-15")
    p.add_argument("--aleatoire", action="store_true", help="ordre aléatoire au lieu de chronologique")
    p.add_argument("--revoir", action="store_true", help="inclure ses images déjà annotées")
    a = p.parse_args()
    ann = Annotateur(a.nom, jour=a.jour, aleatoire=a.aleatoire, revoir=a.revoir)
    print(f"{len(ann.liste)} images à annoter pour {a.nom}. Enregistrement dans {ann.sortie}")
    ann.demarrer()
    print(f"{len(ann.mes)} images dans {ann.sortie.name}. "
          f"Lancer classification.py pour les reporter dans {CSV.name}.")


if __name__ == "__main__":
    main()
