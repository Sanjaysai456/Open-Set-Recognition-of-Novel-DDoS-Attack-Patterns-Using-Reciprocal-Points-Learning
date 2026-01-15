import sys
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config
        self.data_validation_artifact = data_validation_artifact
        self.schema = read_yaml_file(SCHEMA_FILE_PATH)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)
        return df

    def get_data_transformer_object(self) -> Pipeline:
        return Pipeline(
            steps=[
                
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif, k=20)),
                
            ]
        )

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation started")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

            y_train = train_df[TARGET_COLUMN]
            y_test = test_df[TARGET_COLUMN]

            numerical_cols = self.schema["numerical_columns"]

            X_train = train_df[numerical_cols]
            X_test = test_df[numerical_cols]

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            save_object(
                os.path.join(
                    self.data_transformation_config.data_transformation_dir,
                    "label_encoder.pkl"
                ),
                label_encoder
            )

            preprocessor = self.get_data_transformer_object()

            X_train_transformed = preprocessor.fit_transform(X_train.values, y_train)
            X_test_transformed = preprocessor.transform(X_test.values)

            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                feature_columns=numerical_cols  # ✅ FULL columns
            )

        except Exception as e:
            raise MyException(e, sys)
