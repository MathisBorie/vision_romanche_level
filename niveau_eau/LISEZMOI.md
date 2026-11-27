# Niveau d'eau de la Romanche : modèles prêts à l'emploi

Une seule fonction : on lui donne une photo brute de la caméra, elle renvoie le niveau d'eau.

```python
from niveau_eau import mesurer_niveau

y = mesurer_niveau("photo.jpg", methode=3)   # méthode 1, 2 ou 3
```

`y` est la hauteur de la frontière eau / berge d'en face, **en pixels depuis le haut de l'image**
(plus `y` est petit, plus l'eau est haute). Avec `en_metres=True`, la fonction renvoie la **hauteur d'eau
en mètres** sur l'échelle de Simon :

```python
h = mesurer_niveau("photo.jpg", methode=3, en_metres=True)   # ex. 0.96 (m)
```

L'échelle (traits à 1,0 / 1,5 / 1,8 m relevés par Simon) est interpolée linéairement par morceaux, et
prolongée au-delà des traits extrêmes (moins sûr). Le mouvement de la caméra (jusqu'à ~20 px, soit ~15 cm)
est corrigé automatiquement par corrélation sur le mur (`decalage_camera`). Aucun prétraitement à faire : la fonction recadre,
redimensionne et normalise elle-même. Elle accepte un chemin de fichier, une image PIL ou un tableau
numpy RGB. La documentation complète s'affiche au survol de `mesurer_niveau` dans l'éditeur.

## Les 3 méthodes

| `methode` | Modèle | Erreur moyenne | Médiane | Pire cas | À 10 px près |
|---|---|---|---|---|---|
| 1 | SAM 2.1 sans entraînement (zéro-shot) | 8,9 px | 7,0 px | 574 px | 71 % |
| 2 | SAM 2.1, décodeur affiné sur nos annotations | 6,5 px | 6,0 px | 28 px | 77 % |
| 3 | DINOv2 ViT-S gelé + tête entraînée (**par défaut**) | 6,2 px | 5,3 px | 25 px | 79 % |

Erreurs mesurées **avec cette fonction** sur les 883 photos test (3 semaines jamais vues à
l'entraînement : 22-28 juin, 31 août-6 sept., 5-11 oct.).

- **1. SAM zéro-shot** : SAM segmente l'eau à partir de points fixes (eau en bas, berge en haut) ;
  le niveau est le bord supérieur de l'eau. Aucun apprentissage, mais il peut se tromper de plus de
  100 px sur les photos de nuit.
- **2. SAM affiné** : même SAM, seul le décodeur de masques a été réentraîné pour que le bord du masque
  tombe sur nos annotations.
- **3. DINOv2** : l'encodeur DINOv2 (gelé) décrit la bande où se trouve la rivière ; une petite tête
  (convolutions 1D sur le profil vertical) donne la rangée de la frontière. La plus précise.

## Installation

```bash
pip install -r requirements.txt
python exemple.py exemple.jpg
```

Les poids (250 Mo) ne sont pas dans le dépôt git : au premier appel d'une méthode, ceux qui manquent
dans `poids/` sont téléchargés depuis la [Release GitHub `modeles-v1`](https://github.com/MathisBorie/vision_romanche_level/releases/tag/modeles-v1).

Fonctionne sur GPU NVIDIA (`cuda`), Mac (`mps`) ou CPU : le plus rapide disponible est choisi
automatiquement (`appareil="cpu"` pour forcer). Le premier appel charge le modèle (quelques secondes),
les suivants le réutilisent.

## Contenu

| Fichier | Rôle |
|---|---|
| `niveau_eau.py` | la fonction `mesurer_niveau(image, methode)` |
| `exemple.py`, `exemple.jpg` | exemple sur une photo du 22 juin (niveau annoté : 718 px) |
| `requirements.txt` | dépendances |
| `reference_cadrage.png` | zone du mur de la photo de référence, pour corriger le mouvement de la caméra |
| `poids/sam2.1-hiera-small/` | SAM 2.1 de Meta (méthodes 1 et 2), 147 Mo |
| `poids/sam_decodeur_affine.pt` | décodeur de SAM affiné (méthode 2), 16 Mo |
| `poids/dinov2_niveau.pt` | DINOv2 + tête entraînée (méthode 3), 86 Mo |

## Limites

- Conçu pour **cette caméra** et ce cadrage (photos 1919 x 1753, ou 1920 x 2560 en mode portrait). Une autre taille est remise à
  l'échelle, mais un autre point de vue ne marchera pas.
- Le niveau est dans le repère de l'image : si la caméra bouge de quelques pixels, la valeur bouge
  d'autant.
