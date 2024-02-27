import cv2
import numpy as np

def normalize_images(X, target_size):
    """
    Normalise les images en niveaux de gris, applique un filtre pour supprimer le bruit, détecte les contours pour trouver le crop optimal, et redimensionne les images à target_size.
    
    Args:
    - X: np.array, liste d'images (ndarray)
    - target_size: tuple, taille cible des images (ex: (224, 224))

    Returns:
    - np.array, liste d'images normalisées
    """
    normalized_images = [None] * len(X)

    for i, img in enumerate(X):
        if len(img.shape) == 3:
            # Convertir en niveaux de gris si c'est pas déjà le cas
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Appliquer un filtre pour supprimer le bruit (par exemple, un filtre gaussien)
        denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # Détecter les contours pour trouver le crop optimal
        _, thresh = cv2.threshold(denoised_img, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Trouver le contour avec la plus grande aire
            max_contour = max(contours, key=cv2.contourArea)

            # Obtenir les coordonnées du rectangle englobant
            x, y, w, h = cv2.boundingRect(max_contour)

            # Cropper l'image pour obtenir la région d'intérêt
            cropped_img = img[y:y+h, x:x+w]

            # Redimensionner à target_size (pour s'assurer que toutes les images ont la même taille)
            normalized_images[i] = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
        else:
            # Redimensionner à target_size si aucun contour n'est détecté
            normalized_images[i] = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return np.array(normalized_images)


def normalize_image(img, target_size):
    """
    Normalise une image en niveaux de gris, applique un filtre pour supprimer le bruit, détecte les contours pour trouver le crop optimal, et redimensionne l'image à target_size.

    Args:
    - img: np.array, image (ndarray)
    - target_size: tuple, taille cible de l'image (ex: (224, 224))

    Returns:
    - np.array, image normalisée
    """
    # Convertir en niveaux de gris si ce n'est pas déjà le cas
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Appliquer un filtre pour supprimer le bruit (par exemple, un filtre gaussien)
        denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    else:
        # Appliquer un filtre pour supprimer le bruit (par exemple, un filtre gaussien)
        denoised_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Détecter les contours pour trouver le crop optimal
    _, thresh = cv2.threshold(denoised_img, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Trouver le contour avec la plus grande aire
        max_contour = max(contours, key=cv2.contourArea)

        # Obtenir les coordonnées du rectangle englobant
        x, y, w, h = cv2.boundingRect(max_contour)

        # Cropper l'image pour obtenir la région d'intérêt
        cropped_img = img[y:y+h, x:x+w]

        # Redimensionner à target_size (pour s'assurer que toutes les images ont la même taille)
        normalized_image = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
    else:
        # Redimensionner à target_size si aucun contour n'est détecté
        normalized_image = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return normalized_image