import os
import shutil


def create_dir(directory_path, remove_if_exists=True):
    # Supprimer le répertoire existant s'il existe
    if os.path.exists(directory_path) and remove_if_exists:
        shutil.rmtree(directory_path)

    # Créer le nouveau répertoire
    os.makedirs(directory_path, exist_ok=(not remove_if_exists))
