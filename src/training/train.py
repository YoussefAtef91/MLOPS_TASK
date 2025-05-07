import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import dvc.api
import mlflow
import mlflow.sklearn
import dagshub
from dotenv import load_dotenv
import os

load_dotenv()

def train_data(X_train, y_train, model_params):

    params = {'min_samples_split': [2,3,4], 'min_impurity_decrease':[0.0, 0.1, 0.2]}
    model = DecisionTreeClassifier(
        max_depth=model_params["max_depth"],
        min_samples_leaf=model_params["min_samples_leaf"],
        max_features=model_params["max_features"]
    )
    dagshub.auth.add_app_token(os.getenv("DAGSHUB_TOKEN"))

    dagshub.init(
        repo_name=os.getenv("DAGSHUB_REPO"),
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        mlflow=True
    )
    mlflow.set_experiment("Titanic-GridSearch")

    mlflow.sklearn.autolog()
    grid_search = GridSearchCV(model, params, cv=5)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="DecisionTreeTitanic"
    )

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