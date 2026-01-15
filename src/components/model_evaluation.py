import sys
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import f1_score

from src.constants import TARGET_COLUMN
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)
from src.entity.s3_estimator import Proj1Estimator
from src.entity.config_entity import ModelPusherConfig


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_pusher_config: ModelPusherConfig
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config

        # ❌ DO NOT LOAD selected_features.pkl HERE

    def get_best_model(self) -> Optional[Proj1Estimator]:
        try:
            estimator = Proj1Estimator(
                bucket_name=self.model_pusher_config.bucket_name,
                model_path=self.model_pusher_config.s3_model_key_path
            )
            return estimator if estimator.is_model_present(
                self.model_pusher_config.s3_model_key_path
            ) else None
        except Exception as e:
            raise MyException(e, sys)

    def _evaluate_open_set_f1(self, model, df: pd.DataFrame) -> float:
        # ✅ PASS RAW FEATURES — MyModel handles alignment
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN].apply(lambda x: 0 if x == "BENIGN" else 1)

        preds = model.predict(X)

        y_true, y_pred = [], []
        for yt, yp in zip(y, preds):
            if yp == "UNKNOWN":
                continue
            y_true.append(int(yt))
            y_pred.append(int(yp))

        return f1_score(y_true, y_pred) if y_true else 0.0

    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            logging.info("Starting open-set model evaluation")

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            if "_id" in test_df.columns:
                test_df.drop(columns=["_id"], inplace=True)

            trained_model = load_object(
                self.model_trainer_artifact.trained_model_file_path
            )

            trained_f1 = self._evaluate_open_set_f1(trained_model, test_df)
            logging.info(f"New model F1: {trained_f1}")

            best_model = self.get_best_model()
            best_f1 = 0.0

            if best_model:
                try:
                    logging.info("Evaluating production model from S3")
                    best_f1 = self._evaluate_open_set_f1(best_model, test_df)
                except Exception as e:
                    logging.error(f"Failed to evaluate production model from S3: {e}")
                    logging.info("Assuming new model is better due to production model incompatibility.")
                    best_f1 = 0.0

            return EvaluateModelResponse(
                trained_model_f1_score=trained_f1,
                best_model_f1_score=best_f1,
                is_model_accepted=trained_f1 > best_f1,
                difference=trained_f1 - best_f1
            )

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            response = self.evaluate_model()
            return ModelEvaluationArtifact(
                is_model_accepted=response.is_model_accepted,
                changed_accuracy=response.difference,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )
        except Exception as e:
            raise MyException(e, sys)
