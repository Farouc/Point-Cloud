import numpy as np
import argparse

# Defining the desired rotation

def Rz(teta):
    '''Définit une matrice de rotation autour de l'axe Oz par l'angle theta'''
    return np.array([[np.cos(teta), -np.sin(teta), 0],
                     [np.sin(teta), np.cos(teta), 0],
                     [0, 0, 1]])

def rotate_point_cloud(cloud, rotation_matrix):
    '''Cette fonction applique une rotation à un nuage de points spécifique en utilisant la matrice de rotation donnée'''
    return np.dot(cloud, rotation_matrix)

def augment_with_rotations(train_points, train_labels, number_of_duplicates, random_rotation=True):
    '''Augmente le dataset d'entraînement avec des rotations aléatoires ou spécifiques'''
    augmented_train_points = []
    augmented_train_labels = []

    for i in range(len(train_points)):
        for j in range(number_of_duplicates):
            if random_rotation:
                # Générer un angle de rotation aléatoire entre 0 et 2*pi
                angle = np.random.uniform(0, 2 * np.pi)
            else:
                # Utiliser des angles spécifiques 2*k*pi/n pour k allant de 0 à n-1
                angle = (2 * np.pi * j) / number_of_duplicates
            
            # Générer une matrice de rotation à partir de cet angle
            rotation_matrix = Rz(angle)
            # Appliquer la rotation au nuage de points
            rotated_points = rotate_point_cloud(train_points[i], rotation_matrix)
            # Ajouter le nuage de points rotatif et son étiquette à la liste augmentée
            augmented_train_points.append(rotated_points)
            augmented_train_labels.append(train_labels[i])

    # Convertir les listes en tableaux numpy
    augmented_train_points = np.array(augmented_train_points)
    augmented_train_labels = np.array(augmented_train_labels)

    return augmented_train_points, augmented_train_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmenter les données d'entraînement avec des rotations aléatoires ou spécifiques")
    parser.add_argument("input_file", type=str, help="Chemin vers le fichier d'entraînement numpy existant")
    parser.add_argument("output_file", type=str, help="Nom et chemin du fichier de sortie pour les données augmentées")
    parser.add_argument("num_duplicates", type=int, default=4, help="Nombre de duplications pour chaque nuage de points")
    parser.add_argument("--random_rotation", action="store_true", help="Utiliser des rotations aléatoires au lieu d'angles spécifiques")
    args = parser.parse_args()

    # Charger les données d'entraînement existantes
    train_data = np.load(args.input_file)
    train_points = train_data["train_points"]
    train_labels = train_data["train_labels"]

    # Augmenter les données d'entraînement avec des rotations aléatoires ou spécifiques
    augmented_train_points, augmented_train_labels = augment_with_rotations(train_points, train_labels, number_of_duplicates=args.num_duplicates, random_rotation=args.random_rotation)

    # Enregistrer les données augmentées dans un nouveau fichier numpy
    np.savez(args.output_file, train_points=augmented_train_points, train_labels=augmented_train_labels)
