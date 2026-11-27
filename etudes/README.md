# Études mathématiques

Approches sans gros réseau : qu'est-ce qui, dans l'image, porte l'information du niveau ?

| Dossier | Idée | Erreur moyenne | À 20 px près |
|---|---|---|---|
| [`pod/`](pod/) | POD des bandes de photo + petit réseau sur les coefficients (`pod_reseau.ipynb`) | 10,5 px | 90 % |
| [`pls/`](pls/) | PLS : directions choisies pour prédire le niveau, + petit réseau | 10,7 px | 88 % |
| [`fourier/`](fourier/) | Filtres de Fourier contre l'éclairage, passe-bande avant la POD | pas de gain | |
| [`produit_scalaire/`](produit_scalaire/) | Ressemblance avec des photos de niveau connu ou des images « marche » | exploratoire | |
| [`lignes_pixels/`](lignes_pixels/) | Méthodes classiques ligne par ligne (couleur, Canny, texture) | 20,6 px | 81 % |
| [`statistiques/`](statistiques/) | Les pires photos de chaque modèle | | |

Limite commune : ombres et reflets changent les pixels sans changer le niveau, et une base linéaire ne sait
pas les ignorer.
