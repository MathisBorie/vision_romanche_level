import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter

images = sorted(pathlib.Path("data/raw/romanche-01/2026/06/15").glob("*.jpg"))

Y_MIN_RATIO = 0.35
Y_MAX_RATIO = 0.55

SAVGOL_WINDOW = 51    # Lissage du profil (impair)
SAVGOL_POLY = 3       # Ordre du polynôme
TOLERANCE = 15        # Écart max pour confirmer par le step
STEP_WINDOW = 40

def compute_texture_profile(image_bgr):
    """
    Calcule un profil basé sur la TEXTURE (variance locale) et non la luminosité.
    L'eau (lisse) aura une texture basse, la berge (rocheuse) aura une texture haute.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Lissage pour enlever le bruit du capteur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calcul de la variance locale (Écart-type) avec une fenêtre (ex: 21x21)
    k = 21
    kernel = np.ones((k, k), np.float32) / (k * k)
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_sq = cv2.filter2D(gray ** 2, -1, kernel)
    local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
    
    # Le profil 1D est la moyenne de cette texture sur chaque ligne
    texture_profile = np.mean(local_std, axis=1)
    
    # Lissage Savitzky-Golay
    profile_smoothed = savgol_filter(texture_profile, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY)
    
    return profile_smoothed, local_std

def method_gradient(profile, y_min, y_max):
    # Dérivée via Savitzky-Golay
    grad = np.abs(savgol_filter(profile, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY, deriv=1))
    grad_s = cv2.GaussianBlur(grad.reshape(-1, 1), (1, 21), 0).flatten()
    idx = np.argmax(grad_s[y_min:y_max])
    return y_min + idx, grad_s

def method_step(profile, y_min, y_max, window):
    n = len(profile)
    step = np.zeros(n, dtype=np.float32)
    for y in range(window, n - window):
        above = np.mean(profile[y - window:y])
        below = np.mean(profile[y:y + window])
        step[y] = abs(below - above)
    step_s = cv2.GaussianBlur(step.reshape(-1, 1), (1, 15), 0).flatten()
    idx = np.argmax(step_s[y_min:y_max])
    return y_min + idx, step_s

if __name__ == "__main__":
    output_pdf = pathlib.Path("Lignes_pixels_texture.pdf")

    with PdfPages(output_pdf) as pdf:
        for image_path in images:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width, _ = image_rgb.shape
            y_min = int(height * Y_MIN_RATIO)
            y_max = int(height * Y_MAX_RATIO)

            # --- PIPELINE TEXTURE ---
            profile, texture_map = compute_texture_profile(image_bgr)

            # --- DÉTECTION ---
            y_grad, sig_grad = method_gradient(profile, y_min, y_max)
            y_step, sig_step = method_step(profile, y_min, y_max, STEP_WINDOW)
        
            # --- FUSION ---
            ecart = abs(y_step - y_grad)
            if ecart <= TOLERANCE:
                y_final, methode = y_grad, "gradient (confirmé par step)"
                cv_color = (255, 255, 255) # Blanc
                mpl_color = "#FFFFFF"
            else:
                y_final, methode = y_grad, "gradient"
                cv_color = (0, 165, 255)   # Orange
                mpl_color = "#FFA500"

            print(f"{image_path.name} : grad={y_grad} step={y_step} → {y_final} ({methode}, écart={ecart})")

            # --- AFFICHAGE ---
            result = image_rgb.copy()
            cv2.line(result, (0, y_grad), (width-1, y_grad), (0, 0, 255), 2)
            cv2.line(result, (0, y_step), (width-1, y_step), (0, 255, 0), 2)
            cv2.line(result, (0, y_final), (width-1, y_final), cv_color, 4)

            fig, axes = plt.subplots(2, 3, figsize=(22, 10), constrained_layout=True)

            axes[0, 0].imshow(result)
            axes[0, 0].set_title(
                f"{image_path.name}\n"
                f"final={y_final} via {methode}  écart={ecart}px",
                fontsize=10
            )
            axes[0, 0].axis("off")

            # Affichage de la carte de texture (les zones rugueuses ressortent)
            axes[0, 1].imshow(texture_map, cmap="inferno")
            axes[0, 1].set_title("Carte de Texture (Variance locale)")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), cmap="gray")
            axes[0, 2].set_title("Image originale (gris)")
            axes[0, 2].axis("off")

            # Graphique du profil de texture
            axes[1, 0].plot(profile, color="purple")
            axes[1, 0].axvline(y_grad, color="red", linestyle="--", label=f"grad={y_grad}")
            axes[1, 0].axvline(y_step, color="green", linestyle="--", label=f"step={y_step}")
            axes[1, 0].axvline(y_final, color=mpl_color, linestyle="-", linewidth=2,
                              label=f"final={y_final}")
            axes[1, 0].axvspan(y_min, y_max, alpha=0.1, color="gray")
            axes[1, 0].set_title("Profil de Texture (Savitzky-Golay)")
            axes[1, 0].legend(fontsize=8)
            axes[1, 0].grid()

            axes[1, 1].plot(sig_grad, color="red")
            axes[1, 1].axvline(y_grad, color="red", linestyle="--")
            axes[1, 1].axvline(y_step, color="green", linestyle=":")
            axes[1, 1].axvline(y_final, color=mpl_color, linestyle="-", linewidth=2)
            axes[1, 1].axvspan(y_min, y_max, alpha=0.1, color="gray")
            axes[1, 1].set_title("Gradient absolu (Dérivée SG)")
            axes[1, 1].grid()

            axes[1, 2].plot(sig_step, color="green")
            axes[1, 2].axvline(y_grad, color="red", linestyle=":")
            axes[1, 2].axvline(y_step, color="green", linestyle="--")
            axes[1, 2].axvline(y_final, color=mpl_color, linestyle="-", linewidth=2)
            axes[1, 2].axvspan(y_min, y_max, alpha=0.1, color="gray")
            axes[1, 2].set_title(f"Step detector (fenêtre={STEP_WINDOW}px)")
            axes[1, 2].grid()

            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nRapport créé : {output_pdf.resolve()}")