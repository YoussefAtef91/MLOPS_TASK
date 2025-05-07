import mlflow
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

MLFLOW_TRACKING_URI = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO')}.mlflow"
MODEL_NAME = "DecisionTreeTitanic"
MODEL_ALIAS = "3"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = mlflow.sklearn.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_ALIAS}")


example_X = np.array([[3, 1.0, 50.0, 0, 0, 531.0, 8.05, 146.0, 3.0]]) 

y_pred = model.predict(example_X)
label_map = {0: "Didn't Survive", 1: "Survived"}
text_pred = label_map.get(int(y_pred[0]), "Unknown")

print(f"Prediction: {y_pred[0]} ({text_pred})")