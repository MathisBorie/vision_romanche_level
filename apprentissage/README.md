# Apprentissage

Réseaux entraînés sur les photos annotées (train 2 652, validation 841, test 883).

| Dossier | Méthode | Erreur moyenne | À 20 px près |
|---|---|---|---|
| [`encodeurs/`](encodeurs/) | DINOv2 gelé + petite tête | 5,9 px | 99,9 % |
| | ResNet-50 / ConvNeXt affinés | 6,3 / 6,5 px | 99,4 / 99,9 % |
| [`sam/`](sam/) | SAM sans entraînement / décodeur affiné | 8,9 / 6,5 px | 95 / 99,4 % |
| [`cnn/`](cnn/) | Petit CNN entraîné de zéro | 7,7 px | 97 % |
| [`comparaison/`](comparaison/) | Toutes les méthodes sur les mêmes photos test | | |

`exporter_poids.py` envoie les poids vers [`niveau_eau/`](../niveau_eau/) ; `verifier_niveau_eau.py` vérifie
le résultat sur le test.
