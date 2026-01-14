import sys
import numpy as np
import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging


class TargetValueMapping:
    def __init__(self):
        self.yes: int = 0
        self.no: int = 1

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class MyModel:
    def __init__(self, preprocessing_object, trained_model_object, centroids, threshold=5.0):
        """
        preprocessing_object : fitted sklearn Pipeline
        trained_model_object : trained classifier
        centroids            : class centroids (RPL)
        threshold            : distance threshold for UNKNOWN detection
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.centroids = centroids
        self.threshold = threshold

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Open-set prediction:
        1. Transform input
        2. Check distance to centroids
        3. If far → UNKNOWN
        4. Else → classifier prediction
        """
        try:
            logging.info("Starting open-set prediction process")

            # Transform input
            X = self.preprocessing_object.transform(dataframe)

            final_predictions = []

            for x in X:
                # Compute distances to centroids
                distances = [
                    np.linalg.norm(x - self.centroids[c])
                    for c in self.centroids
                ]

                # UNKNOWN detection
                if min(distances) > self.threshold:
                    final_predictions.append("UNKNOWN")
                else:
                    pred = self.trained_model_object.predict([x])[0]
                    final_predictions.append(pred)

            return final_predictions

        except Exception as e:
            logging.error("Error occurred in MyModel.predict", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}(OpenSetEnabled)"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}(OpenSetEnabled)"
