from src.data_processing import DataProcessing
from src.model_training import ModelTraining
import mlflow


if __name__ == "__main__":
    data_processor = DataProcessing("artifacts/raw/data.csv")
    data_processor.load_data()
    data_processor.handle_outliers("SepalWidthCm")
    data_processor.split_data()

    mlflow.set_experiment('model_training_experiment')

    with mlflow.start_run():  # Start a new run for tracking this experiment
        trainer = ModelTraining()
        
        # Log hyperparameters
        mlflow.log_param("criterion", "gini")
        mlflow.log_param("max_depth", 30)
        
        # Load data
        X_train, X_test, y_train, y_test = trainer.load_data()
        
        # Train the model
        trainer.train_model(X_train, y_train)
        
        # Evaluate the model
        trainer.evaluate_model(X_test, y_test)

        # Log the trained model
        mlflow.sklearn.log_model(trainer.model, "decision_tree_model")