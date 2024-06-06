# train.py
import numpy as np
import argparse
from nn import build_pointnet
import keras

#  this code is executed like this: 
#  python train.py ./data_numpy/train_data.npz ./data_numpy/test_data.npz

def train_model(train_data_path, test_data_path,num_epochs):
    # Charger les données
    train_data = np.load(train_data_path)
    train_points = train_data["train_points"]
    train_labels = train_data["train_labels"]

    test_data = np.load(test_data_path)
    test_points = test_data["test_points"]
    test_labels = test_data["test_labels"]

    # Construire le modèle
    model = build_pointnet()

    # Compiler le modèle
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )

    # Entraîner le modèle
    model.fit(train_points, train_labels, epochs=num_epochs, validation_data=(test_points, test_labels))

    # Enregistrer les poids d'entraînement
    model.save_weights("pointnet_weights.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PointNet model")
    parser.add_argument("train_data_path", type=str, help="Chemin vers les données d'entraînement")
    parser.add_argument("test_data_path", type=str, help="Chemin vers les données de test")
    parser.add_argument("number of epochs", type=int, help="Nombre d'épochs d'entrainement ")
    args = parser.parse_args()

    train_model(args.train_data_path, args.test_data_path)
