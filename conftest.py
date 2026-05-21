import sys
import os

# Ajoute la racine du projet au PYTHONPATH pour que pytest trouve `src` et `api`
sys.path.insert(0, os.path.dirname(__file__))
