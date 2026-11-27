# Études mathématiques

Approches sans gros réseau de neurones : on cherche ce qui, dans l'image, porte l'information du niveau.
Toutes travaillent sur la bande de la photo où se trouve la rive (y ≈ 500-1100).

| Dossier | Idée | Résultat (erreur moyenne sur le test) |
|---|---|---|
| [`pod/`](pod/) | POD (décomposition en modes propres) des bandes normalisées, puis petit réseau sur les K premiers coefficients. Version finale : `pod_reseau.ipynb` | **10,5 px** avec K = 750 ; en couleur 11,0 px |
| [`pls/`](pls/) | PLS : comme la POD, mais les directions sont choisies pour prédire le niveau | 12,9 px seule, 10,7 px + réseau |
| [`fourier/`](fourier/) | Filtres de Fourier pour enlever l'éclairage, FFT ligne par ligne, passe-bande avant la POD | pas de gain : un bord d'ombre a la même taille que la rive |
| [`produit_scalaire/`](produit_scalaire/) | Ressemblance (produit scalaire) avec des photos de niveau connu, puis avec des images « marche » synthétiques | piste exploratoire |
| [`lignes_pixels/`](lignes_pixels/) | Méthodes classiques ligne par ligne : couleur, Canny + gradient, texture | 20,6 px au mieux (texture) |
| [`statistiques/`](statistiques/) | Où les modèles se trompent : pires photos de chaque modèle | la POD bute sur les ombres de juin, SAM brut sur la nuit |

Le réseau sur la POD fait mieux que la POD linéaire (13,5 px), mais reste loin des réseaux convolutifs
([`../apprentissage/`](../apprentissage/)) : les ombres et les reflets changent beaucoup les pixels sans
changer le niveau, et une base linéaire ne sait pas les ignorer.
