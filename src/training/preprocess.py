import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import pickle

def preprocess_data(df):
    df = df.drop(["PassengerId","Name"], axis=1)
    df['Age'] = df['Age'].fillna(-1)
    df = df.fillna('Missing')
    cats = df.select_dtypes(include=["object"]).columns
    encoder = OrdinalEncoder()
    df[cats] = encoder.fit_transform(df[cats])
    with open(f"models/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    with open("data/processed/y_train.npy", "w") as f:
        y_train.to_csv(f, index=False)
    with open("data/processed/y_test.npy", "w") as f:
        y_test.to_csv(f, index=False)
    return X_train, X_test, y_train, y_test
