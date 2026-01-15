import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from src.entity.estimator import MyModel


class ModelTrainer:
    def __init__(self, data_transformation_artifact, model_trainer_config):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train, test):
        try:
            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            model = RandomForestClassifier(
                n_estimators=self.model_trainer_config._n_estimators,
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                max_depth=self.model_trainer_config._max_depth,
                bootstrap=self.model_trainer_config._bootstrap,
                max_features=self.model_trainer_config._max_features,
                random_state=self.model_trainer_config._random_state,
                n_jobs=-1
            )

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1_score(y_test, y_pred),
                precision_score=precision_score(y_test, y_pred),
                recall_score=recall_score(y_test, y_pred)
            )

            # 🔑 Centroids for open-set recognition
            centroids = {
                c: x_train[y_train == c].mean(axis=0)
                for c in np.unique(y_train)
            }

            return model, metric_artifact, centroids

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            trained_model, metric_artifact, centroids = (
                self.get_model_object_and_report(train_arr, test_arr)
            )

            preprocessing_obj = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            # ✅ Wrap everything inside custom estimator
            my_model = MyModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=trained_model,
                centroids=centroids,
                feature_columns=self.data_transformation_artifact.feature_columns
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                my_model
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

        except Exception as e:
            raise MyException(e, sys)
