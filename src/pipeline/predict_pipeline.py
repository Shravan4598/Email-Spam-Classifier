import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        """
        Initialize the pipeline class
        """
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.target_encoder_path = os.path.join("artifacts", "target_encoder.pkl")

    def predict(self, messages):
        """
        Predict Spam or Ham for given messages.
        `messages` should be a list of strings.
        Returns a list of labels: 'Spam' or 'Ham'.
        """
        try:
            # Load saved objects
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            target_encoder = load_object(file_path=self.target_encoder_path)

            # Convert input list to DataFrame
            if isinstance(messages, list):
                df = pd.DataFrame({"Message": messages})
            else:
                raise CustomException("Input should be a list of messages", sys)

            # Transform messages using TF-IDF preprocessor
            X_transformed = preprocessor.transform(df["Message"])

            # Predict using trained model
            preds = model.predict(X_transformed)

            # Convert numeric labels back to original ('Ham', 'Spam')
            preds_labels = target_encoder.inverse_transform(preds)

            return preds_labels

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Helper class to wrap a single email message for prediction
    """
    def __init__(self, message: str):
        self.message = message

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame({"Message": [self.message]})
        except Exception as e:
            raise CustomException(e, sys)
