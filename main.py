import pandas as pd
import sys
from src.training.preprocess import preprocess_data
from src.training.train import train_data
from src.training.evaluate import evaluate
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    df = pd.read_csv("data/raw/train.csv")
    
    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        encoder_cfg=cfg.pipeline.encoder,
        encoder_name=cfg.pipeline.encoder_name
    )
    
    model = train_data(X_train, y_train,cfg.pipeline.model)
    print(evaluate(X_test, y_test, model))

if __name__ == "__main__":
    main()