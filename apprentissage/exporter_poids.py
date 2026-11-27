"""Copie les poids entraînés dans niveau_eau/poids/, au format attendu par niveau_eau.py.

    .venv/bin/python apprentissage/encodeurs/entrainer.py --encodeur dinov2   # -> encodeurs/resultats/dinov2/modele.pt
    .venv/bin/python apprentissage/sam/sam_affiner.py --epoques 4            # -> sam/evaluation/sam_affine/decodeur.pt
    .venv/bin/python apprentissage/exporter_poids.py

Les poids publiés sont dans la Release GitHub modeles-v1 (téléchargés automatiquement par niveau_eau.py).
"""
from pathlib import Path
import shutil
import torch

ICI = Path(__file__).resolve().parent
POIDS = ICI.parent / "niveau_eau" / "poids"
POIDS.mkdir(parents=True, exist_ok=True)
decodeur = ICI / "sam" / "evaluation" / "sam_affine" / "decodeur.pt"
if decodeur.exists():
    shutil.copy(decodeur, POIDS / "sam_decodeur_affine.pt")
    print("SAM : décodeur affiné copié")
dino = ICI / "encodeurs" / "resultats" / "dinov2" / "modele.pt"
if dino.exists():
    etat = torch.load(dino, map_location="cpu")
    d = {k.replace("encodeur.reseau.", "reseau.", 1): v.cpu().contiguous() for k, v in etat["modele"].items()}
    torch.save(d, POIDS / "dinov2_niveau.pt")
    print(f"DINOv2 : époque {etat['epoque']}, {len(d)} tenseurs")
