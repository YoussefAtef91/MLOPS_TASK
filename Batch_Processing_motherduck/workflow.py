import pandas as pd
import duckdb
import mlflow.sklearn
import pickle
from prefect import flow, task
import os
from dotenv import load_dotenv

load_dotenv()

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DUCKDB_CONN = f"md:titanic test?motherduck_token={MOTHERDUCK_TOKEN}"
MLFLOW_TRACKING_URI = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO')}.mlflow"
MODEL_NAME = "DecisionTreeTitanic"
MODEL_ALIAS = "3"
PREDICTION_TABLE = "predictions"

@task
def extract() -> pd.DataFrame:
    con = duckdb.connect(DUCKDB_CONN)
    query = "SELECT * FROM test;"
    df = con.execute(query).fetchdf()
    return df


@task
def transform(df: pd.DataFrame) -> pd.DataFrame:
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    df = df.drop(columns=["Name", "PassengerId"])
    df[["Sex", "Ticket", "Cabin", "Embarked"]] = df[["Sex", "Ticket", "Cabin", "Embarked"]].fillna("Missing")
    df = df.fillna(-1)
    df[["Sex", "Ticket", "Cabin", "Embarked"]] = encoder.transform(df[["Sex", "Ticket", "Cabin", "Embarked"]])
    return df

@task
def predict(df: pd.DataFrame) -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_ALIAS}")
    features = df.drop(columns=['id'], errors='ignore')
    predictions = model.predict(features)
    df['prediction'] = predictions
    return df


@task
def load(df: pd.DataFrame):
    con = duckdb.connect(DUCKDB_CONN)
    con.execute(f"CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE} AS SELECT * FROM df LIMIT 0")
    con.register('df', df)
    con.execute(f"INSERT INTO {PREDICTION_TABLE} SELECT * FROM df")
    print(f"✅ Predictions written to {PREDICTION_TABLE}")


@flow(name="Titanic Batch Job")
def titanic_batch_job():
    df = extract()
    df = transform(df)
    preds = predict(df)
    load(preds)


if __name__ == "__main__":
    titanic_batch_job()