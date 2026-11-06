# vision_romanche_level

Projet de vision pour le niveau de la Romanche.

## Accès aux images (bucket S3)

Les photos de la caméra sont dans le bucket AWS S3 `romanche-mathis` (région `us-east-2`),
rangées par `romanche-01/AAAA/MM/JJ/HHMMSS_xxxxxxxx_s.jpg` (une image toutes les 15 min).

1. Installer l'environnement :
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
2. Copier `.env.example` en `.env` et y coller la clé d'accès (demander à Mathis, par message privé).
   `.env` est ignoré par git : ne jamais commiter la clé.
3. Tester : `.venv/bin/python bucket.py` (liste les images des dernières 24 h).
4. Explorer : `notebooks/navigation_bucket.ipynb` (lister les dossiers, afficher, télécharger dans `data/raw/`).

```python
from bucket import Bucket
b = Bucket()
dossiers, images = b.ls("romanche-01/2026/11/")   # naviguer dans l'arborescence
img = b.load_image(images[0]["Key"])               # image PIL en mémoire
b.download(b.list_images(since_hours=24))           # télécharger dans data/raw/
```
