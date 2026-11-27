"""Exemple : mesure du niveau d'eau avec les 3 méthodes sur une photo."""
import sys
from niveau_eau import METHODES, mesurer_niveau

photo = sys.argv[1] if len(sys.argv) > 1 else "exemple.jpg"
for m in (1, 2, 3):
    print(f"{m}. {METHODES[m]:48s} {mesurer_niveau(photo, m):7.1f} px")
