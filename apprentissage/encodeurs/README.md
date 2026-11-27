# Encodeur : détecter le niveau d'eau avec un réseau de neurones

But : à partir d'une photo, trouver la **frontière eau / berge d'en face** (le niveau d'eau, `niveau_px`)
avec un réseau de neurones construit sur un **encodeur pré-entraîné**. Contrainte : pas de Segment Anything
(SAM) ni de ses dérivés (SAM 2, MobileSAM, EfficientSAM, HQ-SAM).

## Choix de l'encodeur

Les annotations donnent **un `y` par photo** (la ligne eau / berge), pas un masque eau / roche pixel par pixel.
L'encodeur alimente donc une petite tête qui prédit cette ligne, pas un décodeur de segmentation complet.

| Encodeur (`--encodeur`) | Modèle (timm) | Pré-entraînement | Utilisation |
|---|---|---|---|
| **`dinov2`** (recommandé) | `vit_small_patch14_dinov2.lvd142m`, 22 M paramètres | auto-supervisé (Meta, LVD-142M) | **gelé** : seule la tête apprend, peu de risque de sur-apprentissage avec ~2 400 photos |
| `dinov2-b` | `vit_base_patch14_dinov2.lvd142m`, 86 M | idem | gelé, plus gros |
| `convnext` | `convnext_tiny.fb_in22k_ft_in1k` (3 premiers étages) | ImageNet-22k puis 1k | affiné (taux d'apprentissage faible) |
| `resnet50` | `resnet50.a1_in1k` (3 premiers étages) | ImageNet-1k | affiné, la référence classique |

Autres pistes non codées : SegFormer (encodeur MiT pré-entraîné sur ADE20K, qui contient déjà les classes
eau / rivière / roche), EfficientNet / MobileNetV3 (légers).

## Architecture

```
bande y 300-1350 de la photo (952 x 518 px)
  -> encodeur                      carte (C, Hf, Wf)   pas de 14 px (DINOv2) ou 16 px (CNN)
  -> moyenne sur la largeur        profil vertical (C, Hf)
  -> GroupNorm + 3 convolutions 1D une valeur par rangée, suréchantillonnée x 8
  -> softmax                       probabilité que la frontière soit à chaque rangée
  -> moyenne pondérée (soft-argmax) y prédit, plus fin que la taille d'une case
```

La frontière est presque horizontale : moyenner sur la largeur donne un profil vertical, et la tête n'a qu'à
repérer la rangée où l'image passe de « berge » à « eau ». Perte : smooth L1 en pixels.

**Augmentations** (entraînement seulement) : luminosité, contraste, saturation, teinte ; miroir gauche-droite ;
**décalage vertical de la bande de ± 40 px** avec le niveau décalé d'autant (la caméra bouge jusqu'à ~20 px).

## Données

- Photos annotées : `donnees/annotations/niveau_*.csv`.
- Découpage : `donnees/split_train_test.csv` (semaines entières, 4 semaines train / 1 semaine test).
- **Validation** prise dans train (une semaine de train sur quatre) pour choisir la meilleure époque ;
  le **test** ne sert qu'à l'évaluation finale.
- Cible : `niveau_px` en coordonnées de l'image (ce que le réseau voit). La correction `- decalage_y` ne sert
  qu'ensuite, pour comparer des niveaux entre photos.

## Utilisation

Installation (une fois) : `.venv/bin/pip install torch torchvision timm` (GPU conseillé ; testé sur RTX 3080 Ti).

```bash
.venv/bin/python apprentissage/encodeurs/entrainer.py --encodeur dinov2                          # ~15 min sur GPU
.venv/bin/python apprentissage/encodeurs/entrainer.py --encodeur convnext
.venv/bin/python apprentissage/encodeurs/entrainer.py --encodeur resnet50
.venv/bin/python apprentissage/encodeurs/entrainer.py --encodeur dinov2 --limite 64 --epoques 1  # essai rapide
```

Résultats dans `apprentissage/encodeurs/resultats/<encodeur>/` :
`metriques.json` (erreurs val / test, référence naïve), `historique.csv` (par époque),
`predictions_test.csv` (`fichier, niveau_px, prediction_px, erreur_px`), `modele.pt` (poids, ignoré par git).

## Fichiers

| Fichier | Rôle |
|---|---|
| `donnees.py` | lecture des annotations et du découpage, découpe de la bande, augmentations |
| `modele.py` | encodeurs disponibles et tête (profil vertical + soft-argmax) |
| `entrainer.py` | entraînement, choix de la meilleure époque sur val, évaluation sur test |

## Résultats

21 novembre, 4 376 photos annotées (Aymen, Mathis, Tristan) : train 2 652, val 841, test 883
(3 semaines jamais vues : 22-28 juin, 31 août-6 sept., 5-11 oct.). 20 époques, meilleure époque choisie sur val.

| Test (883 photos) | **DINOv2 ViT-S (gelé)** | ConvNeXt-Tiny (affiné) | ResNet-50 (affiné) | Référence naïve* |
|---|---|---|---|---|
| erreur moyenne | **6,0 px** | 6,5 px | 6,3 px | 47,4 px |
| erreur médiane | **5,3 px** | 5,9 px | 5,3 px | — |
| erreur max | 23,3 px | 25,1 px | **23,0 px** | — |
| à 5 px près | **48,5 %** | 43,4 % | 47,3 % | — |
| à 10 px près | **80,3 %** | 76,9 % | 77,9 % | — |
| à 20 px près | **99,9 %** | 99,9 % | 99,4 % | — |
| meilleure époque | 6 | 2 | 5 | — |
| durée (RTX 3080 Ti) | **9 min** | 12 min | 12 min | — |

\* toujours prédire le niveau moyen de train.

- Les trois encodeurs sont **proches (6 à 6,5 px)** : l'écart entre eux est faible devant l'erreur elle-même.
  DINOv2 est légèrement devant, et c'est le plus simple (encodeur gelé, seule la tête apprend) et le plus rapide.
- Les meilleures époques arrivent tôt (2 à 6) et l'erreur d'entraînement descend à 1,5-3 px alors que la
  validation reste vers 7,5-8 px : la limite vient sans doute en partie de la **précision des annotations**
  (deux personnes ne placent pas la ligne exactement au même endroit). Exemple : avec ResNet-50, l'erreur
  moyenne sur le test va de 4,6 px (photos annotées par Tristan) à 9,0 px (par Mathis).
- Cas difficiles : images avec reflets (≈ 7,5 px au lieu de 6) ; images voilées (peu nombreuses) jusqu'à
  11-14 px avec DINOv2 et ResNet-50.
