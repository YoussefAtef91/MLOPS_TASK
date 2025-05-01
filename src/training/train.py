import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle
from hydra.utils import instantiate


def train_data(X_train, y_train, model_cfg):
    model = DecisionTreeClassifier(
        max_depth=model_cfg.max_depth,
        min_samples_leaf=model_cfg.min_samples_leaf,
        max_features=model_cfg.max_features
    )
    model.fit(X_train, y_train)
    with open(f"models/{model_cfg.model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    return model
