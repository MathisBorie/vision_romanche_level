
import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================
# 1. PARAMÈTRES
# ==========================================

images = sorted(pathlib.Path("data/raw/romanche-01/2026/11/06").glob("*.jpg"))


# Région verticale de recherche
Y_MIN_RATIO = 0.40
Y_MAX_RATIO = 0.60

# Lissage du signal statistique
SMOOTHING = 21


# ==========================================
# 2. FONCTION DE DÉTECTION
# ==========================================

def detect_waterline(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape

    # Calcul des statistiques RGB
    means = np.mean(image, axis=1)
    stds = np.std(image.astype(np.float32), axis=1)

    # Différences entre lignes successives
    diff_mean = np.linalg.norm(np.diff(means, axis=0), axis=1)
    diff_std = np.linalg.norm(np.diff(stds, axis=0), axis=1)

    # Score statistique
    score = (diff_mean + diff_std) / 255.0

    # Lissage
    score_smooth = cv2.GaussianBlur(
        score.astype(np.float32).reshape(-1, 1),
        (1, SMOOTHING),
        0
    ).flatten()

    # Région de recherche
    y_min = int(height * Y_MIN_RATIO)
    y_max = int(height * Y_MAX_RATIO)

    # Recherche du maximum
    local_index = np.argmax(score_smooth[y_min:y_max])

    waterline_y = y_min + local_index

    # Image avec ligne détectée
    result = image.copy()

    cv2.line(
        result,
        (0, waterline_y),
        (width - 1, waterline_y),
        (255, 0, 0),
        4
    )

    return image, result, score_smooth, waterline_y


# ==========================================
# 3. TRAITEMENT ET CRÉATION DU RAPPORT PDF
# ==========================================

output_pdf = pathlib.Path("Lignes_pixel2_report.pdf")

if not images:
    raise FileNotFoundError("Aucune image trouvée dans data/raw/romanche-01/2026/11/06")

with PdfPages(output_pdf) as pdf:
    for image_path in images:
        image, result, score, waterline = detect_waterline(image_path)

        print(f"\nImage : {image_path}")
        print(f"Niveau détecté : {waterline} pixels")
        print(f"Position relative : {waterline/image.shape[0]*100:.2f} %")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

        axes[0].imshow(image)
        axes[0].set_title(f"{image_path.name} — Image originale")
        axes[0].axis("off")

        axes[1].imshow(result)
        axes[1].set_title(f"Ligne eau / berge détectée — y={waterline} px")
        axes[1].axis("off")

        axes[2].plot(score)
        axes[2].axvline(
            waterline,
            color="red",
            linestyle="--",
            label="Ligne détectée",
        )
        axes[2].set_xlabel("Position verticale (pixels)")
        axes[2].set_ylabel("Score statistique")
        axes[2].set_title("Variation des statistiques RGB")
        axes[2].legend()
        axes[2].grid()

        pdf.savefig(fig)
        plt.close(fig)

print(f"\nRapport créé : {output_pdf.resolve()}")
