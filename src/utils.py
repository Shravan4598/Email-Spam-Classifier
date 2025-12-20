import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save a Python object to the specified file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object from a pickle file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple classification models with hyperparameter tuning using GridSearchCV.
    Returns a dictionary of model name and test F1 score.
    """
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_f1 = f1_score(y_test, y_test_pred)
            report[model_name] = test_f1

        return report

    except Exception as e:
        raise CustomException(e, sys)

def classification_metrics(y_true, y_pred):
    """
    Compute common classification metrics and return as a dictionary.
    """
    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred)
        }
        return metrics

    except Exception as e:
        raise CustomException(e, sys)
