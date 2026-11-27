# niveau_eau

Mesure le niveau d'eau sur une photo brute de la caméra.

```python
from niveau_eau import mesurer_niveau

mesurer_niveau("photo.jpg")                    # niveau en px depuis le haut de l'image (DINOv2)
mesurer_niveau("photo.jpg", en_metres=True)    # hauteur d'eau en m
mesurer_niveau("photo.jpg", methode=1)         # 1 = SAM brut, 2 = SAM affiné, 3 = DINOv2
```

Installation : `pip install -r requirements.txt`. Les poids (250 Mo) se téléchargent tout seuls au premier
appel depuis la [Release](https://github.com/MathisBorie/vision_romanche_level/releases/tag/modeles-v1).

| `methode` | Modèle | Erreur moyenne | Pire cas | À 10 px près | À 20 px près |
|---|---|---|---|---|---|
| 1 | SAM sans entraînement | 8,9 px | 574 px | 71 % | 95,0 % |
| 2 | SAM affiné | 6,5 px | 28 px | 77 % | 99,4 % |
| 3 | DINOv2 (par défaut) | 6,2 px | 25 px | 79 % | 99,9 % |

Mesuré sur 883 photos jamais vues à l'entraînement. Les mètres viennent de repères à 1,0 / 1,5 / 1,8 m
(interpolés entre eux, extrapolés au-delà) ; le mouvement de la caméra est corrigé.
