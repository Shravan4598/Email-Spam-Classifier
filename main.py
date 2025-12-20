from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

if __name__ == "__main__":
    logging.info("==== Starting Spam/Ham Classification Pipeline ====")

    # Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion("Notebook/mail_data.csv")

    # Data Transformation
    transformer = DataTransformation()
    train_arr, test_arr, _, _ = transformer.initiate_data_transformation(train_path, test_path)

    # Split features and target
    X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
    X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

    # Model Training
    trainer = ModelTrainer()
    train_metrics, test_metrics = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

    # Print final metrics
    print("\n===== Final Model Evaluation Report =====")
    print("------ Train Metrics ------")
    print(f"Accuracy  : {train_metrics['accuracy']:.4f}")
    print(f"Precision : {train_metrics['precision']:.4f}")
    print(f"Recall    : {train_metrics['recall']:.4f}")
    print(f"F1 Score  : {train_metrics['f1_score']:.4f}")

    print("\n------ Test Metrics ------")
    print(f"Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"Precision : {test_metrics['precision']:.4f}")
    print(f"Recall    : {test_metrics['recall']:.4f}")
    print(f"F1 Score  : {test_metrics['f1_score']:.4f}")

    logging.info("==== Spam/Ham Classification Pipeline Completed ====")
