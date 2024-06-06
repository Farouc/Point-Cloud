import numpy as np
import argparse
from augment_rot import augment_with_rotations
from augment_noise import augment_with_noise
from augment_outliers import augment_with_outliers

def apply_augmentation(train_points, train_labels,
                       number_of_duplicates_rotation=0, random_rotation=True,
                       number_of_duplicates_noise=0, random_sigma=True,
                       number_of_duplicates_outliers=0, random_number_outliers=True):
    '''Applique les augmentations spécifiées aux données d'entraînement'''
    augmented_train_points, augmented_train_labels = [], []

    # Augmenter avec rotations
    if number_of_duplicates_rotation > 0:
        rot_points = augment_with_rotations(train_points,train_labels, number_of_duplicates_rotation, random_rotation)[0]
        augmented_train_points.extend(rot_points)
        augmented_train_labels.extend([label for label in train_labels for _ in range(number_of_duplicates_rotation)])
    
    # Augmenter avec bruit
    if number_of_duplicates_noise > 0:
        noise_points = augment_with_noise(train_points, train_labels,number_of_duplicates_noise, random_sigma)[0]
        augmented_train_points.extend(noise_points)
        augmented_train_labels.extend([label for label in train_labels for _ in range(number_of_duplicates_noise)])
    
    # Augmenter avec points aberrants
    if number_of_duplicates_outliers > 0:
        outlier_points = augment_with_outliers(train_points, train_labels,number_of_duplicates_outliers, random_number_outliers)[0]
        augmented_train_points.extend(outlier_points)
        augmented_train_labels.extend([label for label in train_labels for _ in range(number_of_duplicates_outliers)])
    
    return np.array(augmented_train_points), np.array(augmented_train_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmenter les données d'entraînement avec différentes techniques")
    parser.add_argument("input_file", type=str, help="Chemin vers le fichier d'entraînement numpy existant")
    parser.add_argument("output_file", type=str, help="Nom et chemin du fichier de sortie pour les données augmentées")
    parser.add_argument("--num_duplicates_rotation", type=int, default=0, help="Nombre de duplications pour les rotations")
    parser.add_argument("--random_rotation", action="store_true", help="Utiliser des rotations aléatoires")
    parser.add_argument("--num_duplicates_noise", type=int, default=0, help="Nombre de duplications pour le bruit")
    parser.add_argument("--random_sigma", action="store_true", help="Utiliser un écart-type aléatoire pour le bruit")
    parser.add_argument("--num_duplicates_outliers", type=int, default=0, help="Nombre de duplications pour les points aberrants")
    parser.add_argument("--random_number_outliers", action="store_true", help="Utiliser un nombre aléatoire de points aberrants")

    args = parser.parse_args()

    # Charger les données d'entraînement existantes
    train_data = np.load(args.input_file)
    train_points = train_data["train_points"]
    train_labels = train_data["train_labels"]

    # Appliquer les augmentations
    augmented_train_points, augmented_train_labels = apply_augmentation(
        train_points, train_labels,
        number_of_duplicates_rotation=args.num_duplicates_rotation, random_rotation=args.random_rotation,
        number_of_duplicates_noise=args.num_duplicates_noise, random_sigma=args.random_sigma,
        number_of_duplicates_outliers=args.num_duplicates_outliers, random_number_outliers=args.random_number_outliers
    )

    # Enregistrer les données augmentées dans un nouveau fichier numpy
    np.savez(args.output_file, train_points=augmented_train_points, train_labels=augmented_train_labels)
