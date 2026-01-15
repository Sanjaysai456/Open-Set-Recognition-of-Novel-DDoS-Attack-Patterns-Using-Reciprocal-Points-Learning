import pandas as pd
from src.entity.s3_estimator import Proj1Estimator
from src.entity.config_entity import ModelPusherConfig

class DDoSPredictor:
    def __init__(self):
        config = ModelPusherConfig()
        self.estimator = Proj1Estimator(
            bucket_name=config.bucket_name,
            model_path=config.s3_model_key_path
        )

    def predict(self, dataframe: pd.DataFrame):
        if "Label" in dataframe.columns:
            dataframe = dataframe.drop(columns=["Label"])
        return self.estimator.predict(dataframe)
