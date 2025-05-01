import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import pickle
import dvc.api


def preprocess_data(df, encoder_name):
    df = df.drop(["PassengerId", "Name"], axis=1)
    df["Age"] = df["Age"].fillna(-1)
    df = df.fillna("Missing")

    cats = df.select_dtypes(include=["object"]).columns

    encoder = OrdinalEncoder()

    df[cats] = encoder.fit_transform(df[cats])

    with open(f"models/{encoder_name}.pkl", "wb") as f:
        pickle.dump(encoder, f)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy", y_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    df = pd.read_csv("data/raw/train.csv")
    encoder_name = "encoder"

    preprocess_data(df, encoder_name)