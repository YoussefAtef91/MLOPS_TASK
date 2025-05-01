import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report


def evaluate(X_test, y_test, model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = np.load("data/processed/y_test.npy")

    model_path = "models/model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    evaluate(X_test, y_test, model)