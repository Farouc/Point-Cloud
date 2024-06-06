import numpy as np
import argparse
import tensorflow as tf

def augment_with_noise(train_points, train_labels, number_of_duplicates, random_number_outliers=True):
    '''Augmente le dataset d'entraînement avec des rotations aléatoires ou spécifiques'''
    augmented_train_points = []
    augmented_train_labels = []

    for i in range(len(train_points)):
        for j in range(number_of_duplicates):
            if random_number_outliers:
                # Générer un bruit d'écart type aléatoire entre 0 et 2
                number_outliers = np.random.randint(0, 20)
            else:
                # Utiliser des bruits spécifiques 20*k/n pour k allant de 0 à n-1
                number_outliers = int(( 20 * j) / number_of_duplicates)
            
            # Appliquer la rotation au nuage de points
            num_points = train_points[i].shape[0]
            indices = tf.range(num_points)
            shuffled_indices = tf.random.shuffle(indices)
            
            selected_indices = shuffled_indices[number_outliers:]
            augmnented_points = tf.gather(train_points[i], selected_indices, axis=0)

            additional_points = tf.random.uniform((number_outliers, 3),
                                                tf.reduce_min(augmnented_points),
                                                tf.reduce_max(augmnented_points),
                                                dtype=tf.float64)
            augmnented_points = tf.concat([augmnented_points, additional_points], axis=0) 
            # Ajouter le nuage de points rotatif et son étiquette à la liste augmentée
            augmented_train_points.append(augmnented_points)
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
    parser.add_argument("--random_number_outliers", action="store_true", help="Utiliser des rotations aléatoires au lieu d'angles spécifiques")
    args = parser.parse_args()

    # Charger les données d'entraînement existantes
    train_data = np.load(args.input_file)
    train_points = train_data["train_points"]
    train_labels = train_data["train_labels"]

    # Augmenter les données d'entraînement avec des rotations aléatoires ou spécifiques
    augmented_train_points, augmented_train_labels = augment_with_noise(train_points, train_labels, number_of_duplicates=args.num_duplicates, random_number_outliers=args.random_number_outliers)

    # Enregistrer les données augmentées dans un nouveau fichier numpy
    np.savez(args.output_file, train_points=augmented_train_points, train_labels=augmented_train_labels)