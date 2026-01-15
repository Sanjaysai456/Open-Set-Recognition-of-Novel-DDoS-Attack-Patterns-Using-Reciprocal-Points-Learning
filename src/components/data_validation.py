import json
import sys
import os
import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH, TARGET_COLUMN


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            df = pd.read_csv(file_path)
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
                logging.info("Dropped _id column")
            return df
        except Exception as e:
            raise MyException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            dataframe_columns = set(df.columns)

            # ✅ ONLY validate numerical columns
            missing_numerical = [
                col for col in self._schema_config["numerical_columns"]
                if col not in dataframe_columns
            ]

            if missing_numerical:
                logging.error(f"Missing numerical columns: {missing_numerical}")

            if TARGET_COLUMN not in dataframe_columns:
                logging.error(f"Target column {TARGET_COLUMN} missing")
                return False

            return not missing_numerical

        except Exception as e:
            raise MyException(e, sys)

    def _reorder_columns(self, df: DataFrame) -> DataFrame:
        ordered_cols = self._schema_config["numerical_columns"] + [TARGET_COLUMN]
        return df[ordered_cols]

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            error_msg = ""

            if not self.is_column_exist(train_df):
                error_msg += "Missing required numerical columns in training data. "

            if not self.is_column_exist(test_df):
                error_msg += "Missing required numerical columns in test data. "

            if error_msg == "":
                train_df = self._reorder_columns(train_df)
                test_df = self._reorder_columns(test_df)

                train_df.to_csv(self.data_ingestion_artifact.trained_file_path, index=False)
                test_df.to_csv(self.data_ingestion_artifact.test_file_path, index=False)

            validation_status = error_msg == ""

            os.makedirs(
                os.path.dirname(self.data_validation_config.validation_report_file_path),
                exist_ok=True
            )

            with open(self.data_validation_config.validation_report_file_path, "w") as f:
                json.dump(
                    {
                        "validation_status": validation_status,
                        "message": error_msg.strip()
                    },
                    f,
                    indent=4
                )

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=error_msg.strip(),
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            logging.info(f"Data validation completed: {artifact}")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
