import pandas as pd
import sys
from src.training.preprocess import preprocess_data
from src.training.train import train_data
from src.training.evaluate import evaluate

def main():
    df = pd.read_csv("data/raw/train.csv")
    
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    model = train_data(X_train, y_train)
    print(evaluate(X_test, y_test, model))

if __name__ == "__main__":
    main()