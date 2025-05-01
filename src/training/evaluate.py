import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


def evaluate(X_test, y_test, model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

