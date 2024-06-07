import numpy as np
import matplotlib.pyplot as plt
from augmentation_modules.augment_rot import augment_with_rotations
from augmentation_modules.augment_noise import augment_with_noise
from augmentation_modules.augment_outliers import augment_with_outliers
from nn import build_pointnet
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def train_model(model, train_points, train_labels, test_points, test_labels, epochs=10, batch_size=32):
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_points, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_points, test_labels), verbose=0)
    val_accuracy = history.history['val_accuracy'][-1]
    return val_accuracy

def apply_augmentation(train_points, train_labels,
                       number_of_duplicates_rotation=0, random_rotation=True,
                       number_of_duplicates_noise=0, random_sigma=True,
                       number_of_duplicates_outliers=0, random_number_outliers=True):
    augmented_train_points, augmented_train_labels = [], []
    if number_of_duplicates_rotation > 0:
        rot_points = augment_with_rotations(train_points, train_labels, number_of_duplicates_rotation, random_rotation)[0]
        augmented_train_points.extend(rot_points)
        augmented_train_labels.extend([label for label in train_labels for _ in range(number_of_duplicates_rotation)])
    if number_of_duplicates_noise > 0:
        noise_points = augment_with_noise(train_points, train_labels, number_of_duplicates_noise, random_sigma)[0]
        augmented_train_points.extend(noise_points)
        augmented_train_labels.extend([label for label in train_labels for _ in range(number_of_duplicates_noise)])
    if number_of_duplicates_outliers > 0:
        outlier_points = augment_with_outliers(train_points, train_labels, number_of_duplicates_outliers, random_number_outliers)[0]
        augmented_train_points.extend(outlier_points)
        augmented_train_labels.extend([label for label in train_labels for _ in range(number_of_duplicates_outliers)])
    return np.array(augmented_train_points), np.array(augmented_train_labels)

# Charger les données d'entraînement et de test
train_data = np.load("./data_numpy/train_data.npz")
train_points = train_data["train_points"]
train_labels = train_data["train_labels"]

test_data = np.load("./data_numpy/test_data.npz")
test_points = test_data["test_points"]
test_labels = test_data["test_labels"]

# Normaliser les étiquettes pour qu'elles soient dans la plage [0, 9]
train_labels = train_labels - 2
test_labels = test_labels - 2

n_duplicates_list = [1, 2, 3, 4, 5]
augmentation_types = ['rotations', 'noise', 'outliers']
accuracies = {aug_type: [] for aug_type in augmentation_types}

# Échantillonner une fraction des données d'entraînement (5% ici)
fraction = 0.05
indices = np.random.choice(train_points.shape[0], size=int(fraction * train_points.shape[0]), replace=False)
train_points_sampled = train_points[indices]
train_labels_sampled = train_labels[indices]

for n in n_duplicates_list:
    for aug_type in augmentation_types:
        print(f"Training with {n} duplicates ({aug_type})...")
        model = build_pointnet()
        
        if aug_type == 'rotations':
            aug_train_points, aug_train_labels = apply_augmentation(train_points_sampled, train_labels_sampled, number_of_duplicates_rotation=n)
        elif aug_type == 'noise':
            aug_train_points, aug_train_labels = apply_augmentation(train_points_sampled, train_labels_sampled, number_of_duplicates_noise=n)
        elif aug_type == 'outliers':
            aug_train_points, aug_train_labels = apply_augmentation(train_points_sampled, train_labels_sampled, number_of_duplicates_outliers=n)
        
        acc = train_model(model, aug_train_points, aug_train_labels, test_points, test_labels, epochs=1, batch_size=32)
        accuracies[aug_type].append(acc)

# Tracer les résultats pour chaque type d'augmentation individuellement
for aug_type in augmentation_types:
    plt.figure(figsize=(10, 5))
    plt.plot(n_duplicates_list, accuracies[aug_type], marker='o', label=f"Augmentation: {aug_type.capitalize()}")
    plt.xlabel("Nombre de duplications")
    plt.ylabel("Accuracy")
    plt.title(f"Impact du nombre de duplications sur l'accuracy ({aug_type.capitalize()})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"augmentation_accuracy_{aug_type}.png")
    plt.close()

# Tracer les résultats globaux regroupant tous les types d'augmentation
plt.figure(figsize=(15, 7))
for aug_type in augmentation_types:
    plt.plot(n_duplicates_list, accuracies[aug_type], marker='o', label=f"Augmentation: {aug_type.capitalize()}")
plt.xlabel("Nombre de duplications")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Impact du nombre de duplications sur l'accuracy")
plt.grid(True)
plt.savefig("augmentation_accuracy_plot.png")
plt.close()

# Augmenter le dataset avec un mix de rotations et de bruit
mixed_accuracies = []
for n in n_duplicates_list:
    print(f"Training with mixed augmentation ({n} duplicates of rotations and noise)...")
    model = build_pointnet()
    aug_train_points, aug_train_labels = apply_augmentation(train_points_sampled, train_labels_sampled, number_of_duplicates_rotation=n, number_of_duplicates_noise=n)
    acc = train_model(model, aug_train_points, aug_train_labels, test_points, test_labels, epochs=1, batch_size=32)
    mixed_accuracies.append(acc)

# Tracer les résultats pour l'augmentation mixte
plt.figure(figsize=(10, 5))
plt.plot(n_duplicates_list, mixed_accuracies, marker='o', label="Mixed Augmentation (Rotations + Noise)")
plt.xlabel("Nombre de duplications")
plt.ylabel("Accuracy")
plt.title("Impact du nombre de duplications sur l'accuracy (Mix Rotations + Noise)")
plt.legend()
plt.grid(True)
plt.savefig("augmentation_accuracy_mixed.png")
plt.close()

print("Training completed and plots saved.")
