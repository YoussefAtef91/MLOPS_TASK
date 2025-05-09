import numpy as np
import litserve as ls
import pickle
import pandas as pd

from src.deployment.online.requests import InferenceRequest

class InferenceAPI(ls.LitAPI):
    def setup(self, device = "cpu"):
        with open("models/model.pkl", "rb") as pkl:
            self._model = pickle.load(pkl)
        with open("models/encoder.pkl", "rb") as pkl:
            self._encoder = pickle.load(pkl)

    def decode_request(self, request):
        InferenceRequest(**request["input"])
        data = [val for val in request["input"].values()]
        data = pd.DataFrame([data], columns=list(request["input"].keys()))
        data = data.drop(columns=["Name", "PassengerId"])
        data[["Sex", "Ticket", "Cabin", "Embarked"]] = self._encoder.transform(data[["Sex", "Ticket", "Cabin", "Embarked"]])
        x = np.asarray(data)
        # x = np.expand_dims(x, 0)
        return x

    def predict(self, x):
        if x is not None:
            return self._model.predict(x)
        else:
            return None

    def encode_response(self, output):
        if output is None:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        return {
            "message": message,
            "prediction": output.tolist()
        }


