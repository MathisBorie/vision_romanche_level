"""Fusionne les annotations brutes du modèle (annotations/raw/*.json) en un CSV chronologique,
et produit une planche de contrôle (data/preview/controle_<nom>.png).

Usage : python merge_annotations.py data/views_top100 annotations/annotations_top100.csv
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from annotation_tools import XS
from bucket import DATA_DIR, ROOT

MIN_FIAB = 2          # colonnes de fiabilité <= 1 écartées du calcul de la moyenne
MIN_COLS = 5          # sous ce nombre de colonnes valides, l'image est marquée peu fiable


def merge(views_dir: Path, out_csv: Path) -> list[dict]:
    index = {v["view"]: v for v in json.load(open(views_dir / "index.json"))}
    ann = {}
    for f in sorted(glob.glob(str(ROOT / "annotations" / "raw" / "*.json"))):
        for a in json.load(open(f)):
            if a["view"] in index:
                ann[a["view"]] = a
    rows = []
    for view, meta in index.items():
        a = ann.get(view)
        r = {"datetime": meta["datetime"], "key": meta["key"], "rang_qualite": meta["rang_qualite"], "view": view}
        if a is None:
            r.update(annotable="", statut="non annoté")
        else:
            ys = a.get("y") or [None] * 10
            fs = a.get("fiabilite_colonnes") or [1] * 10
            valid = [y for y, f in zip(ys, fs) if y is not None and f >= MIN_FIAB]
            r["annotable"] = a["annotable"]
            for x, y, f in zip(XS, ys, fs):
                r[f"y_x{x}"], r[f"f_x{x}"] = y, f
            r["n_valides"] = len(valid)
            r["y_moyen"] = round(sum(valid) / len(valid), 1) if valid else None
            r["fiabilite"] = a.get("fiabilite")
            r["statut"] = ("non annotable" if not a["annotable"] else
                           "ok" if len(valid) >= MIN_COLS else "peu fiable")
            r["commentaire"] = a.get("commentaire", "")
        rows.append(r)
    rows.sort(key=lambda r: r["datetime"])
    fields = (["datetime", "key", "rang_qualite", "view", "annotable", "statut", "y_moyen", "n_valides", "fiabilite"]
              + [f"y_x{x}" for x in XS] + [f"f_x{x}" for x in XS] + ["commentaire"])
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def planche(rows: list[dict], out_png: Path, y0=680, y1=920, cols=4, tw=480):
    """Bande y0-y1 de chaque image avec la frontière annotée (rouge) et y_moyen (jaune)."""
    th = int((y1 - y0) * tw / 1919 * 2) + 16
    sheet = Image.new("RGB", (cols * tw, -(-len(rows) // cols) * th), (20, 20, 20))
    for n, r in enumerate(rows):
        im = Image.open(DATA_DIR / r["key"]).convert("RGB").crop((0, y0, 1919, y1)).resize((1919, (y1 - y0) * 2))
        d = ImageDraw.Draw(im)
        pts = [(x, (r[f"y_x{x}"] - y0) * 2) for x in XS if r.get(f"y_x{x}") is not None]
        if len(pts) > 1:
            d.line(pts, fill=(255, 0, 0), width=6)
        for p in pts:
            d.ellipse([p[0] - 10, p[1] - 10, p[0] + 10, p[1] + 10], fill=(255, 0, 0))
        if r.get("y_moyen") is not None:
            d.line([(0, (r["y_moyen"] - y0) * 2), (1919, (r["y_moyen"] - y0) * 2)], fill=(255, 255, 0), width=3)
        t = im.resize((tw, th - 16))
        x, y = (n % cols) * tw, (n // cols) * th
        sheet.paste(t, (x, y + 16))
        col = {"ok": (120, 220, 90), "peu fiable": (240, 160, 40)}.get(r["statut"], (230, 70, 70))
        ImageDraw.Draw(sheet).text((x + 3, y + 2), f"{r['view']} {r['datetime'][5:16]} {r['statut']} "
                                   f"y={r.get('y_moyen')} f={r.get('fiabilite')}", fill=col)
    sheet.save(out_png)


if __name__ == "__main__":
    views_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
    rows = merge(views_dir, out_csv)
    planche(rows, ROOT / "data" / "preview" / f"controle_{out_csv.stem}.png")
    from collections import Counter
    print(out_csv, Counter(r["statut"] for r in rows))
