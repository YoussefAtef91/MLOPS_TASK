import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle
import dvc.api

def train_data(X_train, y_train, model_params):
    model = DecisionTreeClassifier(
        max_depth=model_params["max_depth"],
        min_samples_leaf=model_params["min_samples_leaf"],
        max_features=model_params["max_features"]
    )
    model.fit(X_train, y_train)

    with open(f"models/{model_params['model_name']}.pkl", "wb") as f:
        pickle.dump(model, f)

    return model


if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = np.load("data/processed/y_train.npy")

    model_params = {
        "model_name": "model",
        "max_depth": 5,
        "min_samples_leaf": 2,
        "max_features": 0.5
    }

    train_data(X_train, y_train, model_params)