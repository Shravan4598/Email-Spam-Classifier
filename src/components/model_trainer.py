import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, classification_metrics

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting Logistic Regression training with GridSearchCV (solver='saga')")

            # Use saga solver for sparse data to remove convergence warnings
            lr = LogisticRegression(max_iter=5000, solver='saga')

            # Hyperparameter grid
            param_grid = {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"]
            }

            # Grid search with F1 scoring
            grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            logging.info(f"Best parameters: {grid.best_params_}")

            # Predictions
            train_pred = best_model.predict(X_train)
            test_pred = best_model.predict(X_test)

            # Metrics
            train_metrics = classification_metrics(y_train, train_pred)
            test_metrics = classification_metrics(y_test, test_pred)

            logging.info(f"Train Metrics: {train_metrics}")
            logging.info(f"Test Metrics: {test_metrics}")

            # Save model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info("Model saved successfully")

            return train_metrics, test_metrics

        except Exception as e:
            raise CustomException(e, sys)
