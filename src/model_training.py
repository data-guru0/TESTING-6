import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ModelTraining:
    def __init__(self):
        self.processed_data_path = "artifacts/processed"
        self.model_path = "artifacts/models"
        os.makedirs(self.model_path, exist_ok=True)
        self.model = DecisionTreeClassifier(criterion="gini", max_depth=30, random_state=42)
    
    def load_data(self):
        X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
        X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
        y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
        y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, os.path.join(self.model_path, "model.pkl"))
        print("Model training complete. Model saved in artifacts/models/")

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        
        precision = precision_score(y_test, y_pred, average='weighted')  # or 'micro' or 'weighted'
        recall = recall_score(y_test, y_pred, average='weighted')  # or 'micro' or 'weighted'
        f2 = f1_score(y_test, y_pred, average='weighted')  # F2 score with beta=2
        print(f"Test Accuracy: {accuracy}")
        print(f"Test Precision: {precision}")
        print(f"Test Recall: {recall}")
        print(f"Test F2 Score: {f2}")

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        confusion_matrix_path = os.path.join(self.model_path, "confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        plt.close()

        # Log metrics and artifacts to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f2)
        mlflow.log_artifact(confusion_matrix_path)

if __name__ == "__main__":
    # Set up MLflow experiment
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