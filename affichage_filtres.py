import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================
# SCRIPT DE VISUALISATION DES FILTRES
# ==========================================

# On prend une image représentative (ou la première du dossier)
# Change ce chemin si tu veux tester une image en particulier
images = sorted(pathlib.Path("data/raw/romanche-01/2026/11/06").glob("*.jpg"))
if not images:
    raise FileNotFoundError("Aucune image trouvée.")
image_path = images[30] # ou images[10] pour en tester une autre

bgr = cv2.imread(str(image_path))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# 1. Préparation des filtres
# ------------------------------------------

# Flou gaussien pour réduire le bruit
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Détection de contours de Canny
canny = cv2.Canny(blurred, 50, 150)

# Gradient de Sobel (détection des variations horizontales)
sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
sobel_x = np.absolute(sobel_x)
sobel_x = np.uint8(np.clip(sobel_x, 0, 255))

# Espace LAB
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# CLAHE sur le canal L (améliore le contraste local)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
L_clahe = clahe.apply(L)

# Espace HSV (parfois utile pour l'eau)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

# 2. Création du rapport PDF
# ------------------------------------------

output_pdf = pathlib.Path("Visualisation_Filtres.pdf")

with PdfPages(output_pdf) as pdf:
    print(f"Génération du rapport pour : {image_path.name}")

    # Page 1 : Image originale vs Niveaux de gris vs Canny
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Niveaux de gris")
    axes[1].axis("off")
    
    axes[2].imshow(canny, cmap="gray")
    axes[2].set_title("Contours de Canny")
    axes[2].axis("off")
    
    fig.suptitle(f"{image_path.name} — Comparaison 1", fontsize=16)
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2 : Gradient vertical (Sobel) vs CLAHE vs Canal L
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    
    axes[0].imshow(sobel_x, cmap="gray")
    axes[0].set_title("Gradient vertical (Sobel Y)")
    axes[0].axis("off")
    
    axes[1].imshow(L_clahe, cmap="gray")
    axes[1].set_title("Canal L + CLAHE (contraste local)")
    axes[1].axis("off")
    
    axes[2].imshow(L, cmap="gray")
    axes[2].set_title("Canal L (Luminosité LAB)")
    axes[2].axis("off")
    
    fig.suptitle(f"{image_path.name} — Comparaison 2", fontsize=16)
    pdf.savefig(fig)
    plt.close(fig)

    # Page 3 : Canaux A et B (Couleurs LAB)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    
    axes[0].imshow(A, cmap="gray")
    axes[0].set_title("Canal A (Vert/Rouge)")
    axes[0].axis("off")
    
    axes[1].imshow(B, cmap="gray")
    axes[1].set_title("Canal B (Bleu/Jaune)")
    axes[1].axis("off")
    
    axes[2].imshow(V, cmap="gray")
    axes[2].set_title("Canal V (HSV)")
    axes[2].axis("off")
    
    fig.suptitle(f"{image_path.name} — Comparaison 3", fontsize=16)
    pdf.savefig(fig)
    plt.close(fig)

print(f"\nRapport créé : {output_pdf.resolve()}")