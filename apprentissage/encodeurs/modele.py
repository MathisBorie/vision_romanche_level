"""Encodeur pré-entraîné + tête qui prédit la rangée de la frontière eau / berge.

    image (bande) -> encodeur -> carte (C, Hf, Wf) -> moyenne sur la largeur -> profil (C, Hf)
        -> convolutions 1D -> une probabilité par rangée -> y = moyenne pondérée des rangées

La frontière est presque horizontale : moyenner sur la largeur donne un profil vertical, et la tête
n'a qu'à repérer la rangée où l'image passe de « berge » à « eau ». La moyenne pondérée (soft-argmax)
donne une position plus fine que la taille d'une case de la carte.
"""
from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn

# Encodeurs disponibles (poids pré-entraînés téléchargés par timm). Segment Anything exclu.
ENCODEURS = {
    "dinov2":   dict(nom="vit_small_patch14_dinov2.lvd142m", vit=True, geler=True),   # 22 M paramètres, gelé
    "dinov2-b": dict(nom="vit_base_patch14_dinov2.lvd142m", vit=True, geler=True),    # 86 M, gelé
    "convnext": dict(nom="convnext_tiny.fb_in22k_ft_in1k", vit=False, indice=2, geler=False),  # affiné
    "resnet50": dict(nom="resnet50.a1_in1k", vit=False, indice=3, geler=False),       # affiné
}


class Encodeur(nn.Module):
    """Renvoie une carte de caractéristiques (B, C, Hf, Wf) au pas de 14 (ViT) ou 16 (CNN)."""

    def __init__(self, cle: str):
        super().__init__()
        cfg = ENCODEURS[cle]
        self.vit, self.gele = cfg["vit"], cfg["geler"]
        if self.vit:
            self.reseau = timm.create_model(cfg["nom"], pretrained=True, dynamic_img_size=True, num_classes=0)
            self.canaux, self.pas = self.reseau.embed_dim, self.reseau.patch_embed.patch_size[0]
        else:
            self.reseau = timm.create_model(cfg["nom"], pretrained=True, features_only=True,
                                            out_indices=[cfg["indice"]])
            self.canaux, self.pas = self.reseau.feature_info.channels()[0], self.reseau.feature_info.reduction()[0]
        if self.gele:
            self.reseau.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.gele:
            self.reseau.eval()               # encodeur gelé : toujours en mode évaluation
        return self

    def forward(self, x):
        if not self.vit:
            return self.reseau(x)[0]
        b, _, h, w = x.shape
        jetons = self.reseau.forward_features(x)[:, self.reseau.num_prefix_tokens:]   # sans le jeton [CLS]
        return jetons.transpose(1, 2).reshape(b, self.canaux, h // self.pas, w // self.pas)


class DetecteurNiveau(nn.Module):
    def __init__(self, cle: str, finesse: int = 8):
        super().__init__()
        self.encodeur = Encodeur(cle)
        c = self.encodeur.canaux
        self.tete = nn.Sequential(
            nn.GroupNorm(1, c),              # met les caractéristiques de tous les encodeurs à la même échelle
            nn.Conv1d(c, 256, 3, padding=1), nn.GELU(),
            nn.Conv1d(256, 256, 3, padding=1), nn.GELU(),
            nn.Conv1d(256, 1, 1),
        )
        self.finesse = finesse               # le profil est suréchantillonné x finesse avant le soft-argmax

    def forward(self, x):
        carte = self.encodeur(x)                         # (B, C, Hf, Wf)
        profil = carte.float().mean(dim=3)               # (B, C, Hf) : moyenne sur la largeur
        logits = self.tete(profil)                       # (B, 1, Hf)
        logits = F.interpolate(logits, scale_factor=self.finesse, mode="linear", align_corners=False)[:, 0]
        p = logits.softmax(dim=1)                        # probabilité que la frontière soit à chaque rangée
        n = p.shape[1]
        centres = (torch.arange(n, device=p.device, dtype=p.dtype) + 0.5) / n
        return (p * centres).sum(dim=1), p               # y relatif dans la bande [0, 1], distribution
