# Apprentissage

Réseaux entraînés sur les photos annotées, tous avec le même découpage : train (2 652 photos), validation
(841, pour choisir l'époque) et test (883, jamais vues : juin, septembre, octobre).

| Dossier | Méthode | Erreur moyenne sur le test |
|---|---|---|
| [`encodeurs/`](encodeurs/) | Encodeur pré-entraîné + petite tête qui repère la rangée de la rive : DINOv2 (gelé), ConvNeXt, ResNet-50 | **5,9 px** (DINOv2), 6,5, 6,3 |
| [`sam/`](sam/) | SAM 2.1 segmente l'eau avec des points fixes ; sans entraînement, ou en réentraînant son décodeur | 8,9 px brut, 6,5 px affiné |
| [`cnn/`](cnn/) | Petit CNN entraîné de zéro (augmentations : ombres, gamma, bruit) | 7,7 px |
| [`comparaison/`](comparaison/) | Relance toutes les méthodes sur la même machine et produit le tableau commun | `comparaison/resultats/` |

`exporter_poids.py` copie les poids entraînés vers [`../niveau_eau/`](../niveau_eau/), et
`verifier_niveau_eau.py` repasse les 883 photos test dans la fonction publiée (résultats dans `verification/`).

```bash
.venv/bin/python apprentissage/encodeurs/entrainer.py --encodeur dinov2      # ~10 min sur GPU
.venv/bin/python apprentissage/sam/sam_affiner.py --epoques 4
.venv/bin/python apprentissage/comparaison/tout_relancer.py                 # tout, ~1 h 15 sur RTX 3080 Ti
```
