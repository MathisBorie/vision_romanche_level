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

## Préparation des données (7 novembre 2026)

Chaîne de traitement, à lancer dans l'ordre (chaque étape peut être relancée : elle ne refait que ce qui manque).

### 1. Télécharger toutes les images

```bash
.venv/bin/python -c "from bucket import Bucket; b = Bucket(); b.download(b.list_images())"
```

Au 7 novembre : 5 527 images (2,7 Go) du 29 mai au 7 novembre 2026, dans `data/raw/` (ignoré par git).

### 2. Rassembler les images dans un seul dossier : `restructure.py`

```bash
.venv/bin/python data_treatment/restructure.py
```

- `data/raw/romanche-01/2026/11/06/104512_02821fd2_s.jpg` → `data/images/2026-11-06-104512.jpg`
- Liens physiques par défaut : pas d'espace disque en plus, `data/raw/` reste intact (`--move` pour déplacer).
- **Images portrait** : les 1er, 3 et 4 juin, la caméra était en mode portrait (1920 × 2560). Même objectif,
  même échelle : elles sont recadrées sur la bande y 807-2560, qui correspond au cadrage paysage (1919 × 1753).
  Les 16 images recadrées portent le commentaire JPEG `recadree-portrait` ; l'original n'est pas modifié.

### 3. Mesurer les images : `standardisation.py` → `data_treatment/catalogue_images.csv`

```bash
.venv/bin/python data_treatment/standardisation.py
```

Lit les images (quelques secondes) et écrit les mesures : luminosité, contraste, netteté, pixels saturés,
et surtout le **décalage de la caméra** par rapport à l'image de référence `2026-11-03-104512.jpg`, qui
permet de ramener toutes les images dans le même repère (voir plus bas). Ne décide rien.

### 4. Classer les images : `classification.py`

```bash
.venv/bin/python data_treatment/classification.py
```

Applique les seuils sur les mesures (période, visibilité, reflets, exploitable). Ne relit pas les images :
instantané, on peut changer les seuils (en haut du script) et relancer. Ils ont été réglés en regardant
des planches d'images.

### Le catalogue `data_treatment/catalogue_images.csv`

Une ligne par image. Le CSV est versionné (dans `data_treatment/`, à côté des scripts) pour partager les annotations.
Relancer `standardisation.py` ou `classification.py` ne touche jamais aux colonnes manuelles
(sauf `classification.py`, qui y reporte les fichiers d'annotation de l'étape 5).

| Colonnes | Écrites par | Contenu |
|---|---|---|
| `fichier`, `datetime`, `jour`, `heure` | standardisation | identification ; **heures en UTC** (heure française = UTC + 1 ou + 2) |
| `format` | standardisation | `paysage` / `recadree` (ancienne portrait) / `autre` |
| `luminosite`, `contraste` | standardisation | gris moyen et écart-type de la bande utile (y 500-1200) |
| `nettete` | standardisation | netteté de la zone fixe (mur + enrochement) |
| `pct_sature` | standardisation | % de pixels blancs saturés (reflets, surexposition) |
| `decalage_x`, `decalage_y`, `confiance_alignement` | standardisation | mouvement de la caméra par rapport à l'image de référence |
| `periode` | classification | `jour` / `crepuscule` / `nuit` (d'après la luminosité) |
| `visibilite` | classification | `nette` / `brumeuse` / `voilee` |
| `reflets` | classification | 1 si `pct_sature` ≥ 2 % |
| `exploitable` | classification | `non` = nuit ; `douteux` = voilée, brumeuse, reflets ou crépuscule (à trier à la main) ; `oui` sinon |
| `niveau_px` | **manuel** | y de la frontière eau / berge d'en face (px, image d'origine) |
| `classe_niveau` | **manuel** | `bas` / `moyen` / `haut` / `crue` (étape suivante) |
| `verifie` | **manuel** | 1 = ligne vérifiée par un humain : `periode`, `visibilite`, `reflets`, `exploitable` ne sont plus recalculés |
| `annotateur` | classification (depuis les fichiers d'annotation) | qui a annoté le niveau |
| `commentaire` | **manuel** | texte libre |

Bilan au 7 novembre : 3 901 images `oui`, 511 `douteux`, 1 115 `non` (nuits).

**La caméra bouge.** Le décalage vertical de la scène est d'environ 20 px en juin et de 0 en novembre (dérive lente),
et varie jusqu'à ~15 px au cours d'une journée de soleil (dilatation du support). Pas de zoom ni de rotation.
Un niveau en pixels ne se compare donc d'une image à l'autre qu'après correction : **`niveau_px - decalage_y`**
(`decalage_y` > 0 : la scène est plus bas que sur la référence).

### 5. Annoter le niveau à la main : `define_water_lvl.py`

Les jours sont répartis entre les trois annotateurs (jour de l'année modulo 3, environ 1 450 images chacun,
sur toute la saison) : personne n'annote deux fois la même image. Chacun lance le script avec son nom :

```bash
.venv/bin/python data_treatment/define_water_lvl.py aymen                  # ou mathis, tristan
.venv/bin/python data_treatment/define_water_lvl.py aymen --jour 2026-09   # seulement septembre
.venv/bin/python data_treatment/define_water_lvl.py aymen --revoir         # inclut ses images déjà annotées
```

**Un clic enregistre** le niveau à la hauteur cliquée (frontière eau / berge d'en face, y moyen sur la largeur,
comme dans le notebook 04) et passe directement à l'image suivante.

| Touche | Action |
|---|---|
| clic gauche | enregistre le niveau et passe à la suivante |
| Entrée / `p` | enregistre la suggestion (pointillés jaunes : dernier niveau annoté, corrigé du mouvement de la caméra) |
| `x` | rejette l'image (`exploitable = non`) et passe à la suivante |
| → | passe sans rien enregistrer |
| ← | revient à l'image précédente (ligne rouge = niveau enregistré ; recliquer pour corriger) |
| `q` | quitter (tout est déjà enregistré ; au prochain lancement, reprise à la première image non annotée) |

Chacun enregistre dans **son propre fichier** `data_treatment/annotations/niveau_<nom>.csv`
(`fichier, niveau_px, exploitable, annote_le`) : pas de conflit git, chacun commite le sien.
Pour reporter toutes les annotations dans le catalogue (`niveau_px`, `exploitable`, `verifie = 1`, `annotateur`) :

```bash
.venv/bin/python data_treatment/classification.py
```

Étape suivante, une fois les images annotées : la classification du niveau (`classe_niveau`).
