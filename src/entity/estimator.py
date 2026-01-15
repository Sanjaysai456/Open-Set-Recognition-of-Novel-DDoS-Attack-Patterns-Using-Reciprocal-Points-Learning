import sys
import numpy as np
import pandas as pd

from src.exception import MyException
from src.logger import logging


class MyModel:
    def __init__(
        self,
        preprocessing_object,
        trained_model_object,
        centroids,
        feature_columns,
        threshold=5.0
    ):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.centroids = centroids
        self.feature_columns = feature_columns  # ALL 77 numerical columns
        self.threshold = threshold

    def predict(self, dataframe: pd.DataFrame):
        try:
            logging.info("Starting prediction")

            # 1️⃣ Reconstruct full feature vector using the index of the input dataframe
            # This ensures we handle batch predictions correctly
            full_df = pd.DataFrame(0.0, index=dataframe.index, columns=self.feature_columns)

            for col in self.feature_columns:
                if col in dataframe.columns:
                    full_df[col] = dataframe[col]

            # 2️⃣ Convert to NumPy (pipeline was fit on NumPy)
            X = full_df.to_numpy(dtype=float)

            # 3️⃣ Apply SAME pipeline (scaler + selector)
            X_transformed = self.preprocessing_object.transform(X)

            # 4️⃣ Model prediction
            preds = self.trained_model_object.predict(X_transformed)

            # 5️⃣ Open-set logic
            final_preds = []
            for x, pred in zip(X_transformed, preds):
                distances = [
                    np.linalg.norm(x - self.centroids[c])
                    for c in self.centroids
                ]

                if min(distances) > self.threshold:
                    final_preds.append("UNKNOWN")
                else:
                    final_preds.append(int(pred))

            return final_preds

        except Exception as e:
            raise MyException(e, sys)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}(OpenSetEnabled)"
