"""Passe les 883 photos test dans niveau_eau/niveau_eau.py et compare aux résultats de la comparaison.

    .venv/bin/python apprentissage/verifier_niveau_eau.py 1 2 3
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "donnees").is_dir())
sys.path.insert(0, str(ROOT / "niveau_eau"))
from niveau_eau import mesurer_niveau, METHODES
import os; APPAREIL = os.environ.get("APPAREIL")
REF = {1: "sam_zero_shot", 2: "sam_affine", 3: "encodeur_dinov2"}
def metr(e):
    a = np.abs(e)
    return dict(moyenne=round(a.mean(), 2), mediane=round(float(np.median(a)), 2), max=round(a.max(), 1),
                biais=round(e.mean(), 2), a5=round(100 * (a <= 5).mean(), 1), a10=round(100 * (a <= 10).mean(), 1),
                a20=round(100 * (a <= 20).mean(), 1))
sortie = Path(__file__).resolve().parent / "verification"
for m in map(int, sys.argv[1:] or [1, 2, 3]):
    ref = pd.read_csv(ROOT / "apprentissage" / "comparaison" / "resultats" / f"{REF[m]}_predictions.csv")
    t0 = time.time()
    ref["nous_px"] = [mesurer_niveau(ROOT / "data/images" / f, m, appareil=APPAREIL) for f in ref.fichier]
    ref["nous_erreur"] = ref.nous_px - ref.niveau_px
    ref.to_csv(sortie / f"verif_methode{m}.csv", index=False)
    r = {"methode": METHODES[m], "Aymen": metr(ref.erreur_px.to_numpy()), "livraison": metr(ref.nous_erreur.to_numpy()),
         "ecart_moyen_entre_predictions_px": round(float((ref.nous_px - ref.prediction_px).abs().mean()), 2),
         "duree_s": round(time.time() - t0)}
    (sortie / f"verif_methode{m}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    print(json.dumps(r, ensure_ascii=False), flush=True)
