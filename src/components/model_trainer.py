import os
import sys

from dataclasses import dataclass


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModeltrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    This function is responsible for model training
    """

    def __init__(self):
        self.model_train_config = ModeltrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train and return model and find best accuracy
        """
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "KNN-Classifier": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "GradientBoostClassifier": GradientBoostingClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
            }

            params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"],
                    "max_iter": [100, 500, 1000],
                },
                "KNN-Classifier": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"],
                    "p": [1, 2],
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [None, "sqrt", "log2"],
                },
                "GradientBoostClassifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "subsample": [0.8, 1.0],
                },
                "RandomForestClassifier": {
                    "n_estimators": [100, 200, 300],
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None],
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_train_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
