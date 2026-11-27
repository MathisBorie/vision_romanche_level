"""Mesure du niveau d'eau de la Romanche sur une photo de la caméra.

Projet INF8422 (Polytechnique Montréal) : Mathis Borie, Aymen Bouakkaz, Tristan.

Utilisation rapide ::

    from niveau_eau import mesurer_niveau

    y = mesurer_niveau("photo.jpg", methode=3)      # DINOv2 (la plus précise)
    print(f"niveau d'eau : {y:.0f} px depuis le haut de l'image")

Le seul point d'entrée à connaître est :func:`mesurer_niveau`. Les poids des modèles sont lus
dans le dossier ``poids/`` à côté de ce fichier. S'ils manquent (dépôt git cloné), ils sont
téléchargés automatiquement depuis la Release GitHub du projet au premier appel (250 Mo en tout).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn

__all__ = ["mesurer_niveau", "en_metres_depuis_px", "decalage_camera", "METHODES", "ECHELLE"]

ICI = Path(__file__).resolve().parent
POIDS = ICI / "poids"
RELEASE = "https://github.com/MathisBorie/vision_romanche_level/releases/download/modeles-v1/"
FICHIERS_SAM = ["sam2.1-hiera-small/config.json", "sam2.1-hiera-small/processor_config.json",
                "sam2.1-hiera-small/model.safetensors"]

#: Numéro de méthode -> nom lisible.
METHODES = {
    1: "SAM 2.1 sans entraînement (zéro-shot)",
    2: "SAM 2.1, décodeur affiné sur nos annotations",
    3: "DINOv2 ViT-S gelé + tête entraînée",
}

# Géométrie de la caméra : toutes les constantes sont en pixels de la photo d'origine (1919 x 1753).
W0, H0 = 1919, 1753
PORTRAIT, RECADRAGE_PORTRAIT = (1920, 2560), (0, 807, 1919, 2560)   # anciennes photos en mode portrait

# SAM : prompt fixe (identique pour toutes les photos)
Y_EAU, Y_BERGE = 1500, 450                      # rangée de points « eau » (positifs) / « berge » (négatifs)
X_POINTS = list(range(200, 1800, 300))
BOITE = [0, 300, 1919, 1753]                    # exclut la végétation du haut
LISSAGE = 31                                    # médiane glissante sur la bordure (méthode 1)
COTE = 1024                                     # entrée carrée de SAM
NETTETE = 16.0                                  # méthode 2 : facteur appliqué aux logits du masque
H_RANG = H0 / 256                               # hauteur d'une rangée du masque 256 x 256
BANDE_SAM = (300, 1350)

# DINOv2 : bande donnée au réseau
Y0, Y1 = 300, 1350
TAILLE = (952, 518)                             # (largeur, hauteur), multiples de 14
MOYENNE, ECART_TYPE = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

ImageEntree = Union[str, Path, Image.Image, np.ndarray]

#: Échelle de Simon : (niveau en px dans le repère de référence, hauteur en m), de haut en bas de l'image.
#: Repère de référence = cadrage de la photo 2026-11-03-104512 (decalage_y = 0, voir decalage_camera).
#: Relevé sur sa capture des photos du 21/11/2026 (affichage y >= 300, échelle 1919/1871, decalage_y ~ 1,5).
ECHELLE = [(696.5, 1.8), (785.7, 1.5), (846.7, 1.0)]
REFERENCE_CADRAGE = ICI / "reference_cadrage.png"   # zone fixe (mur, y < 800) de la référence, à 1/4
S_CADRAGE, FIXE_Y1, LUM_MIN = 4, 800, 20
_CACHE: dict = {}                               # modèles déjà chargés (un seul chargement par session)


# --------------------------------------------------------------------------------------------------
# Point d'entrée
# --------------------------------------------------------------------------------------------------
def mesurer_niveau(image: ImageEntree, methode: Literal[1, 2, 3] = 3,
                   appareil: str | None = None, en_metres: bool = False) -> float:
    """Mesure le niveau d'eau sur une photo de la caméra de la Romanche.

    Le niveau est la hauteur (en pixels, comptée depuis le HAUT de l'image) de la frontière entre
    l'eau et la berge d'en face : plus la valeur est PETITE, plus l'eau est HAUTE.
    La photo est donnée telle quelle, sans recadrage ni prétraitement : la fonction s'en charge.

    Args:
        image: la photo, sous l'une de ces formes :

            - chemin vers un fichier image (``str`` ou ``pathlib.Path``), par exemple
              ``"data/images/2026-06-22-083020.jpg"`` ;
            - image PIL (``PIL.Image.Image``), n'importe quel mode (RGB, L, RGBA...) ;
            - tableau numpy ``(H, W, 3)`` en **RGB** (attention : OpenCV lit en BGR, faire
              ``img[:, :, ::-1]`` avant) ou ``(H, W)`` en niveaux de gris, ``uint8`` 0-255
              ou ``float`` 0-1.

            Format attendu : une photo de la caméra du projet (cadrage fixe, 1919 x 1753 px ;
            les anciennes photos en mode portrait 1920 x 2560 sont aussi acceptées).
            Une photo d'une autre taille est mise à l'échelle et le résultat est rendu dans les
            pixels de l'image fournie, mais le cadrage doit rester celui de la caméra.
        methode: la méthode à utiliser (entier de 1 à 3) :

            - ``1`` : **SAM 2.1 sans entraînement** (zéro-shot). SAM segmente l'eau à partir de
              points fixes ; le niveau est le bord supérieur de l'eau (médiane sur la largeur).
              Test : 8,9 px d'erreur moyenne, mais peut se tromper lourdement la nuit (> 100 px).
            - ``2`` : **SAM 2.1, décodeur affiné** sur nos photos annotées.
              Test : 6,5 px d'erreur moyenne, 28 px au pire.
            - ``3`` : **DINOv2** (encodeur gelé + petite tête entraînée). La plus précise.
              Test : 6,2 px d'erreur moyenne, 25 px au pire. *Méthode par défaut.*

            (Erreurs mesurées sur 883 photos de 3 semaines jamais vues à l'entraînement.)
        appareil: ``"cuda"``, ``"mps"`` (Mac) ou ``"cpu"``. Par défaut, le plus rapide disponible.
        en_metres: si ``True``, renvoie la hauteur d'eau en **mètres** sur l'échelle de Simon
            (traits à 1,0 / 1,5 / 1,8 m, interpolation linéaire par morceaux) au lieu des pixels.
            Le mouvement de la caméra est corrigé automatiquement (voir :func:`decalage_camera`).

    Returns:
        float: le niveau d'eau, en pixels depuis le haut de l'image fournie
        (ou en mètres si ``en_metres=True`` : plus grand = eau plus haute).

    Raises:
        ValueError: si ``methode`` n'est pas 1, 2 ou 3, ou si l'image n'a pas un format reconnu.
        FileNotFoundError: si le chemin de l'image ou un fichier de poids est introuvable.

    Exemples:
        >>> from niveau_eau import mesurer_niveau
        >>> mesurer_niveau("exemple.jpg")              # DINOv2 (niveau annoté à la main : 718)
        709.8
        >>> mesurer_niveau("exemple.jpg", methode=2)   # SAM affiné
        712.9

    Notes:
        - Le premier appel pour une méthode charge le modèle (quelques secondes) ; les appels
          suivants le réutilisent. Compter ensuite ~0,1 à 1 s par photo selon la machine.
        - Le niveau est dans le repère de l'image : si la caméra a bougé de quelques pixels,
          la valeur bouge d'autant.
    """
    if methode not in METHODES:
        raise ValueError(f"methode doit être 1, 2 ou 3 (reçu {methode!r}) : {METHODES}")
    photo, jpeg = _ouvrir(image)
    haut = 0
    if photo.size == PORTRAIT:                               # mode portrait : même vue, plus haute
        photo, jpeg, haut = photo.convert("RGB").crop(RECADRAGE_PORTRAIT), False, RECADRAGE_PORTRAIT[1]
    w, h = photo.size
    if methode == 1:
        y = _sam_zero_shot(photo, _appareil(appareil))
    elif methode == 2:
        y = _sam_affine(photo, jpeg, _appareil(appareil))
    else:
        y = _dinov2(photo, jpeg, _appareil(appareil))
    if en_metres:                                            # repère 1919 x 1753, recalé sur la référence
        return en_metres_depuis_px(float(y) - decalage_camera(image if haut == 0 else photo))
    return round(haut + float(y) * h / H0, 1)                # repère 1919 x 1753 -> repère de l'image


def en_metres_depuis_px(niveau_ref_px: float) -> float:
    """Convertit un niveau en px (repère de référence) en hauteur d'eau en m, sur l'échelle de Simon.

    Linéaire par morceaux entre les traits de :data:`ECHELLE` (1,8 / 1,5 / 1,0 m) ; au-delà des traits
    extrêmes, on prolonge le segment le plus proche (valeur moins sûre).

    Args:
        niveau_ref_px: niveau en px dans le repère de référence, c'est-à-dire
            ``niveau_px - decalage_y`` (``decalage_y`` donné par :func:`decalage_camera`).

    Returns:
        float: hauteur d'eau en mètres, arrondie au cm.
    """
    ys = [p for p, _ in ECHELLE]
    hs = [m for _, m in ECHELLE]
    i = 0 if niveau_ref_px < ys[1] else len(ys) - 2           # segment haut ou segment bas
    for k in range(len(ys) - 1):
        if ys[k] <= niveau_ref_px <= ys[k + 1]:
            i = k
    pente = (hs[i + 1] - hs[i]) / (ys[i + 1] - ys[i])
    return round(hs[i] + pente * (niveau_ref_px - ys[i]), 2)


def decalage_camera(image: ImageEntree, _jpeg: bool | None = None) -> float:
    """Décalage vertical de la caméra (px) par rapport au cadrage de référence.

    La caméra dérive (~20 px entre juin et novembre) et bouge avec le soleil (~15 px dans la journée).
    On le mesure comme ``donnees/standardisation.py`` : corrélation de phase sur la zone fixe
    (mur + enrochement, y < 800) à 1/4 de résolution. ``decalage_y > 0`` : la scène est plus bas que
    sur la référence. Renvoie 0 si l'image est trop sombre pour s'aligner (nuit).

    Args:
        image: la photo (mêmes formats que :func:`mesurer_niveau`).

    Returns:
        float: ``decalage_y`` en px de la photo d'origine.
    """
    import cv2
    if _jpeg is None:
        image, _jpeg = _ouvrir(image)
    im = image
    if _jpeg:
        im.draft("RGB", (W0 // S_CADRAGE, H0 // S_CADRAGE))
        im = im.convert("RGB")
    if im.size != (480, 439):
        im = _complete(im).resize((480, 439), Image.BOX)
    gris = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)
    s = W0 / gris.shape[1]
    bande = gris[round(500 / s): round(1200 / s)]
    if bande.mean() < LUM_MIN:
        return 0.0

    def zone(g):
        z = g - cv2.GaussianBlur(g, (0, 0), 8)
        return z * cv2.createHanningWindow(z.shape[::-1], cv2.CV_32F)

    ref = np.asarray(Image.open(REFERENCE_CADRAGE), dtype=np.float32)
    (_, dy), _ = cv2.phaseCorrelate(zone(ref), zone(gris[: round(FIXE_Y1 / s)]))
    return float(dy * s)


# --------------------------------------------------------------------------------------------------
# Outils communs
# --------------------------------------------------------------------------------------------------
def _appareil(appareil: str | None) -> str:
    if appareil:
        return appareil
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _assurer_poids(*noms: str) -> None:
    """Télécharge depuis la Release GitHub les fichiers de poids absents de ``poids/``."""
    import urllib.request
    for nom in noms:
        dest = POIDS / nom
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = RELEASE + nom.replace("/", "__")
        print(f"niveau_eau : téléchargement de {nom} ...", flush=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)


def _ouvrir(image: ImageEntree) -> tuple[Image.Image, bool]:
    """-> (image PIL, True si c'est un JPEG ouvert depuis un fichier et pas encore décodé)."""
    if isinstance(image, (str, Path)):
        chemin = Path(image)
        if not chemin.exists():
            raise FileNotFoundError(chemin)
        im = Image.open(chemin)
        if im.format == "JPEG" and im.getexif().get(0x0112, 1) == 1:
            return im, True                                  # décodage différé (voir _demi)
        return ImageOps.exif_transpose(im).convert("RGB"), False
    if isinstance(image, Image.Image):
        return image.convert("RGB"), False
    if isinstance(image, np.ndarray):
        a = image
        if a.dtype != np.uint8:
            a = np.clip(a * 255 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        if a.ndim != 3 or a.shape[2] not in (3, 4):
            raise ValueError(f"tableau numpy de forme {a.shape} : attendu (H, W, 3) RGB ou (H, W)")
        return Image.fromarray(a[:, :, :3]), False
    raise ValueError(f"type d'image non reconnu : {type(image)}")


def _complete(im: Image.Image) -> Image.Image:
    """Image pleine résolution dans le repère 1919 x 1753."""
    im = im.convert("RGB")
    return im if im.size == (W0, H0) else im.resize((W0, H0), Image.BILINEAR)


def _demi(im: Image.Image, jpeg: bool) -> Image.Image:
    """Image à demi-résolution, comme à l'entraînement (décodage JPEG réduit de PIL)."""
    if jpeg:
        im.draft("RGB", (W0 // 2, H0 // 2))
        im = im.convert("RGB")
        if abs(im.size[0] * 2 - W0) <= 2 and abs(im.size[1] * 2 - H0) <= 2:
            return im
    return _complete(im).reduce(2)


def _normaliser(im: Image.Image) -> torch.Tensor:
    a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    m = torch.tensor(MOYENNE)[:, None, None]
    s = torch.tensor(ECART_TYPE)[:, None, None]
    return ((a - m) / s)[None]


# --------------------------------------------------------------------------------------------------
# Méthodes 1 et 2 : SAM 2.1
# --------------------------------------------------------------------------------------------------
def _charger_sam(appareil: str, affine: bool):
    cle = ("sam", affine, appareil)
    if cle not in _CACHE:
        from transformers import Sam2Model, Sam2Processor
        _assurer_poids(*FICHIERS_SAM, *(["sam_decodeur_affine.pt"] if affine else []))
        dossier = POIDS / "sam2.1-hiera-small"
        modele = Sam2Model.from_pretrained(dossier)
        if affine:
            modele.mask_decoder.load_state_dict(torch.load(POIDS / "sam_decodeur_affine.pt", map_location="cpu"))
        _CACHE[cle] = (Sam2Processor.from_pretrained(dossier), modele.to(appareil).eval())
    return _CACHE[cle]


@torch.no_grad()
def _sam_zero_shot(photo: Image.Image, appareil: str) -> float:
    processeur, modele = _charger_sam(appareil, affine=False)
    im = _complete(photo)
    points = [[x, Y_EAU] for x in X_POINTS] + [[x, Y_BERGE] for x in X_POINTS]
    labels = [1] * len(X_POINTS) + [0] * len(X_POINTS)
    entree = processeur(images=im, input_points=[[points]], input_labels=[[labels]],
                        input_boxes=[[BOITE]], return_tensors="pt").to(appareil)
    sortie = modele(**entree, multimask_output=True)
    masques = processeur.post_process_masks(sortie.pred_masks.cpu(), entree["original_sizes"])[0]
    k = int(sortie.iou_scores.cpu()[0, 0].argmax())          # masque le plus sûr selon SAM
    masque = masques[0, k].numpy().astype(bool)

    # bord supérieur de la plus grande zone d'eau, colonne par colonne, puis médiane
    import cv2
    import pandas as pd
    n, lab, stats, _ = cv2.connectedComponentsWithStats(masque.astype(np.uint8), connectivity=8)
    if n <= 1:
        return float("nan")                                  # aucune eau trouvée
    eau = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    y = np.where(eau.any(axis=0), eau.argmax(axis=0), np.nan).astype(float)
    y = pd.Series(y).rolling(LISSAGE, center=True, min_periods=1).median().to_numpy()
    return float(np.nanmedian(y))


@torch.no_grad()
def _sam_affine(photo: Image.Image, jpeg: bool, appareil: str) -> float:
    _, modele = _charger_sam(appareil, affine=True)
    pixels = _normaliser(_demi(photo, jpeg).resize((COTE, COTE), Image.BILINEAR)).to(appareil)
    emb = modele.get_image_embeddings(pixels)
    sx, sy = COTE / W0, COTE / H0
    pts = [[x * sx, Y_EAU * sy] for x in X_POINTS] + [[x * sx, Y_BERGE * sy] for x in X_POINTS]
    lab = [1] * len(X_POINTS) + [0] * len(X_POINTS)
    boite = [BOITE[0] * sx, BOITE[1] * sy, BOITE[2] * sx, BOITE[3] * sy]
    t = lambda v, dt: torch.tensor([v], dtype=dt, device=appareil)
    sortie = modele(image_embeddings=emb, input_points=t([pts], torch.float32), input_labels=t([lab], torch.long),
                    input_boxes=t([boite], torch.float32), multimask_output=False)
    logits = sortie.pred_masks[:, 0, 0]                      # (1, 256, 256)
    # position attendue de la première rangée d'eau dans la bande, moyennée sur la largeur
    r0, r1 = int(BANDE_SAM[0] // H_RANG), int(np.ceil(BANDE_SAM[1] / H_RANG))
    survie = torch.exp(torch.cumsum(F.logsigmoid(-NETTETE * logits[:, r0:r1, :].float()), dim=1))
    return float(r0 * H_RANG + survie.sum(dim=1).mean(dim=1) * H_RANG)


# --------------------------------------------------------------------------------------------------
# Méthode 3 : DINOv2 + tête « profil vertical »
# --------------------------------------------------------------------------------------------------
class _DetecteurNiveau(nn.Module):
    """DINOv2 ViT-S/14 -> carte de caractéristiques -> moyenne sur la largeur -> convolutions 1D
    -> probabilité de la frontière pour chaque rangée -> y = moyenne pondérée (soft-argmax)."""

    def __init__(self, finesse: int = 8):
        super().__init__()
        import timm
        self.reseau = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False,
                                        dynamic_img_size=True, num_classes=0)
        c = self.reseau.embed_dim
        self.tete = nn.Sequential(nn.GroupNorm(1, c), nn.Conv1d(c, 256, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(256, 256, 3, padding=1), nn.GELU(), nn.Conv1d(256, 1, 1))
        self.finesse = finesse

    def forward(self, x):
        b, _, h, w = x.shape
        jetons = self.reseau.forward_features(x)[:, self.reseau.num_prefix_tokens:]
        carte = jetons.transpose(1, 2).reshape(b, -1, h // 14, w // 14)
        logits = self.tete(carte.float().mean(dim=3))
        logits = F.interpolate(logits, scale_factor=self.finesse, mode="linear", align_corners=False)[:, 0]
        p = logits.softmax(dim=1)
        n = p.shape[1]
        centres = (torch.arange(n, device=p.device, dtype=p.dtype) + 0.5) / n
        return (p * centres).sum(dim=1)                      # y relatif dans la bande, entre 0 et 1


def _charger_dinov2(appareil: str):
    cle = ("dinov2", appareil)
    if cle not in _CACHE:
        _assurer_poids("dinov2_niveau.pt")
        fichier = POIDS / "dinov2_niveau.pt"
        modele = _DetecteurNiveau()
        modele.load_state_dict(torch.load(fichier, map_location="cpu"))
        _CACHE[cle] = modele.to(appareil).eval()
    return _CACHE[cle]


@torch.no_grad()
def _dinov2(photo: Image.Image, jpeg: bool, appareil: str) -> float:
    modele = _charger_dinov2(appareil)
    im = _demi(photo, jpeg)
    s = W0 / im.size[0]
    bande = im.crop((0, round(Y0 / s), im.size[0], round(Y1 / s))).resize(TAILLE, Image.BILINEAR)
    y_rel = modele(_normaliser(bande).to(appareil))
    return Y0 + float(y_rel) * (Y1 - Y0)


if __name__ == "__main__":                                   # python niveau_eau.py photo.jpg [méthode]
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage : python niveau_eau.py photo.jpg [1|2|3]")
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(f"{METHODES[m]} : niveau {mesurer_niveau(sys.argv[1], m):.1f} px")
