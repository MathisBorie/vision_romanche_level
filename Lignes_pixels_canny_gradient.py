import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================
# 1. PARAMÈTRES
# ==========================================

images = sorted(pathlib.Path("data/raw/romanche-01/2026/11/06").glob("*.jpg"))

# Zone verticale de recherche (on garde ta zone)
Y_MIN_RATIO = 0.30
Y_MAX_RATIO = 0.65

# Lissage du signal (un peu plus grand pour bien lisser les micro-variations des rochers)
SMOOTHING = 41

# Paramètres de Canny
CANNY_LOW = 50
CANNY_HIGH = 150


# ==========================================
# 2. FONCTION DE DÉTECTION
# ==========================================

def detect_waterline(image_path):

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise FileNotFoundError(image_path)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape

    # 1. Préparation pour Canny
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # 2. Morphologie horizontale (relie les contours)
    kernel_horizontal = np.ones((1, 15), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_horizontal)

    # 3. Profil de densité de contours par ligne
    profile = np.sum(edges_closed, axis=1).astype(np.float32) / 255.0

    # 4. Lissage du profil
    profile_smooth = cv2.GaussianBlur(
        profile.reshape(-1, 1),
        (1, SMOOTHING),
        0
    ).flatten()

    # 5. LA CLÉ DU PROBLÈME : Calcul du gradient (dérivée)
    # On cherche là où la densité de contours CHUTE (gradient négatif).
    # On prend l'opposé pour avoir un pic positif au niveau de la chute.
    gradient = -np.gradient(profile_smooth)
    
    # On ajoute un petit lissage sur le gradient pour éviter les pics isolés
    gradient_smooth = cv2.GaussianBlur(
        gradient.reshape(-1, 1),
        (1, 21),
        0
    ).flatten()

    # 6. Région de recherche
    y_min = int(height * Y_MIN_RATIO)
    y_max = int(height * Y_MAX_RATIO)

    # 7. Recherche du maximum de la CHUTE de contours
    local_index = np.argmax(gradient_smooth[y_min:y_max])
    waterline_y = y_min + local_index

    # Image avec ligne détectée
    result = image_rgb.copy()
    cv2.line(
        result,
        (0, waterline_y),
        (width - 1, waterline_y),
        (255, 0, 0),
        4
    )

    return image_rgb, result, edges_closed, profile_smooth, gradient_smooth, waterline_y, y_min, y_max


# ==========================================
# 3. TRAITEMENT ET CRÉATION DU RAPPORT PDF
# ==========================================

output_pdf = pathlib.Path("Lignes_pixels_canny_gradient_report.pdf")

if not images:
    raise FileNotFoundError("Aucune image trouvée dans data/raw/romanche-01/2026/11/06")

with PdfPages(output_pdf) as pdf:
    for image_path in images:
        image, result, edges, profile, gradient, waterline, y_min, y_max = detect_waterline(image_path)

        print(f"\nImage : {image_path.name}")
        print(f"Niveau détecté : {waterline} pixels")
        print(f"Position relative : {waterline/image.shape[0]*100:.2f} %")

        # Création d'une figure à 4 panneaux (on ajoute le gradient)
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

        # Panneau 1 : Image avec ligne
        axes[0].imshow(result)
        axes[0].set_title(f"{image_path.name} — Ligne (y={waterline})")
        axes[0].axis("off")

        # Panneau 2 : Contours
        axes[1].imshow(edges, cmap="gray")
        axes[1].set_title("Contours (Canny)")
        axes[1].axis("off")

        # Panneau 3 : Profil de densité (ce qu'on avait avant)
        axes[2].plot(profile)
        axes[2].axvline(waterline, color="red", linestyle="--")
        axes[2].set_title("Densité de contours par ligne")
        axes[2].grid()

        # Panneau 4 : Le GRADIENT (la chute de contours = le vrai bord)
        axes[3].plot(gradient)
        axes[3].axvline(waterline, color="red", linestyle="--", label="Ligne détectée")
        axes[3].axvspan(y_min, y_max, alpha=0.1, color="gray", label="Zone de recherche")
        axes[3].set_title("Gradient (Chute de contours = Bord de l'eau)")
        axes[3].legend()
        axes[3].grid()

        pdf.savefig(fig)
        plt.close(fig)

print(f"\nRapport créé : {output_pdf.resolve()}")