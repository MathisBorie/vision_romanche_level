# Données

Photos de la caméra toutes les 15 min depuis fin mai 2026 (bucket S3), environ 5 500 photos de 1919 × 1753.

| Fichier | Rôle |
|---|---|
| `bucket.py` | accès aux photos (clé dans `.env`, voir `.env.example`) |
| `restructure.py` | renomme les photos `AAAA-MM-JJ-HHMMSS.jpg` |
| `standardisation.py` | luminosité, netteté, mouvement de la caméra → `catalogue_images.csv` |
| `classification.py` | jour / nuit, photo exploitable ou non |
| `define_water_lvl.py` | annotation du niveau à la souris |
| `split_train_test.py` | découpage par semaines : 4 en train, 1 en test |
| `annotations/` | 4 410 photos annotées à la main (Aymen, Mathis, Tristan) |

La caméra bouge (jusqu'à ~20 px) : un niveau se compare d'une photo à l'autre avec `niveau_px - decalage_y`.
