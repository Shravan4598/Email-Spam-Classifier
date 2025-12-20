import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    target_encoder_file_path = os.path.join('artifacts', "target_encoder.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            text_pipeline = Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer())
                ]
            )
            return text_pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data read successfully")

            preprocessing_obj = self.get_data_transformer_object()

            # Encode target
            target_encoder = LabelEncoder()
            y_train = target_encoder.fit_transform(train_df['Category'])
            y_test = target_encoder.transform(test_df['Category'])

            # Transform text
            X_train = preprocessing_obj.fit_transform(train_df['Message'])
            X_test = preprocessing_obj.transform(test_df['Message'])

            # Convert sparse to dense
            X_train_arr = X_train.toarray()
            X_test_arr = X_test.toarray()

            train_arr = np.c_[X_train_arr, y_train]
            test_arr = np.c_[X_test_arr, y_test]

            # Save preprocessor and target encoder
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)
            save_object(self.data_transformation_config.target_encoder_file_path, target_encoder)

            logging.info("Data transformation completed")

            return train_arr, test_arr, \
                   self.data_transformation_config.preprocessor_obj_file_path, \
                   self.data_transformation_config.target_encoder_file_path

        except Exception as e:
            raise CustomException(e, sys)
