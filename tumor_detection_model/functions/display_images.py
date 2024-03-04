import math
import matplotlib.pyplot as plt


def display_images(X, y, n):
    # Filtrer les images par classe
    yes_images = X[y == 1][:n]
    no_images = X[y == 0][:n]

    # Calculer le nombre de lignes et colonnes pour chaque classe
    n_rows = int(math.sqrt(n))
    n_cols = math.ceil(n / n_rows)

    # Créer la figure pour la classe "yes"
    fig_yes, axs_yes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    fig_yes.suptitle("Class: Yes")

    # Afficher des images de la classe "yes"
    for i in range(min(n, len(yes_images))):
        axs_yes[i // n_cols, i % n_cols].imshow(yes_images[i])

    # Créer la figure pour la classe "no"
    fig_no, axs_no = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    fig_no.suptitle("Class: No")

    # Afficher des images de la classe "no"
    for i in range(min(n, len(no_images))):
        axs_no[i // n_cols, i % n_cols].imshow(no_images[i])

    plt.show()
