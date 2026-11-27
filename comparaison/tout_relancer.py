"""Relance TOUTES les méthodes sur ce PC, puis produit le tableau comparatif.

    .venv/bin/python comparaison/tout_relancer.py                          # tout (~1 h 15 sur RTX 3080 Ti)
    .venv/bin/python comparaison/tout_relancer.py --methodes lignes pod     # seulement certaines étapes
    .venv/bin/python comparaison/tout_relancer.py --methodes tableau        # redessiner le tableau seulement

Étapes (dans l'ordre) :
    lignes      Stats RGB, Canny + gradient, Texture          comparaison/methode_lignes.py      ~10 min, CPU
    pod         POD + linéaire / réseau (K = 300, 750)         comparaison/methode_pod.py         ~8 min, CPU
    sam         SAM 2.1-S zéro-shot                            comparaison/methode_sam.py         ~3 min, GPU
    sam_affine  SAM 2.1-S, décodeur affiné (4 époques)         comparaison/methode_sam.py         ~20 min, GPU
    encodeurs   ResNet-50, ConvNeXt-Tiny, DINOv2 (réentraînés) comparaison/methode_encodeurs.py  ~35 min, GPU
    tableau     tableau_comparaison.png / .csv                 comparaison/tableau.py

Chaque étape tourne dans son propre processus ; son journal est dans comparaison/resultats/journal_<étape>.txt.
Si une étape échoue, les suivantes continuent et le tableau est fait avec ce qui est disponible.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ICI = Path(__file__).resolve().parent
RESULTATS = ICI / "resultats"
ETAPES = {
    "lignes": ["methode_lignes.py"],
    "pod": ["methode_pod.py"],
    "sam": ["methode_sam.py", "--zero-shot"],
    "sam_affine": ["methode_sam.py", "--affine"],
    "encodeurs": ["methode_encodeurs.py"],
    "tableau": ["tableau.py"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methodes", nargs="+", choices=list(ETAPES), default=list(ETAPES),
                    help="étapes à lancer (défaut : toutes ; le tableau est toujours refait à la fin)")
    a = ap.parse_args()
    etapes = [e for e in ETAPES if e in a.methodes and e != "tableau"] + ["tableau"]
    RESULTATS.mkdir(parents=True, exist_ok=True)

    debut, bilan = time.time(), []
    for e in etapes:
        journal = RESULTATS / f"journal_{e}.txt"
        print(f"[{time.strftime('%H:%M')}] {e} ... (journal : {journal.relative_to(ICI.parent)})", flush=True)
        t0 = time.time()
        with open(journal, "w") as j:
            code = subprocess.run([sys.executable, "-u", str(ICI / ETAPES[e][0]), *ETAPES[e][1:]],
                                  stdout=j, stderr=subprocess.STDOUT, cwd=ICI.parent).returncode
        statut = "ok" if code == 0 else f"ÉCHEC (code {code}, voir le journal)"
        bilan.append((e, statut, time.time() - t0))
        print(f"           {statut} en {(time.time() - t0) / 60:.1f} min", flush=True)
        # résumé de l'étape (lignes de résultats du journal)
        for ligne in journal.read_text(errors="replace").splitlines():
            if "erreur moyenne" in ligne and "|" in ligne:
                print("          " + ligne.strip())

    print(f"\nTerminé en {(time.time() - debut) / 60:.0f} min")
    for e, statut, d in bilan:
        print(f"  {e:11s} {statut:10s} {d / 60:5.1f} min")
    print(f"Tableau : {(RESULTATS / 'tableau_comparaison.png').relative_to(ICI.parent)}")


if __name__ == "__main__":
    main()
