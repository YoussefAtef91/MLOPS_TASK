import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle


def train_data(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    with open(f"models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model
