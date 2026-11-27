# Encodeurs

Un encodeur pré-entraîné décrit la bande de la photo où se trouve la rive ; on moyenne sur la largeur et une
petite tête (convolutions 1D) donne la rangée de la frontière eau / berge.

| Encodeur | Erreur moyenne | Pire cas | À 10 px près | À 20 px près |
|---|---|---|---|---|
| DINOv2 ViT-S (gelé) | 5,9 px | 23 px | 83 % | 99,9 % |
| ResNet-50 (affiné) | 6,3 px | 23 px | 78 % | 99,4 % |
| ConvNeXt-Tiny (affiné) | 6,5 px | 25 px | 77 % | 99,9 % |

```bash
python apprentissage/encodeurs/entrainer.py --encodeur dinov2
```
