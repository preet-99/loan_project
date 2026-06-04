import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformer:
    def __init__(self):
        self.data_transformer_config = DataTransformerConfig()

    def get_data_transformer_obj(self):
        """
        This function is responsible for making transformation object
        """
        try:
            df = pd.read_csv("artifacts/train.csv")

            df.drop(columns=["loan_status"], inplace=True)

            numerical_columns = df.select_dtypes(exclude="object").columns
            categorical_columns = df.select_dtypes(include="object").columns

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responsible for doing data transformation
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_obj() ## Get preprocessor object

            target_column_name = "loan_status"  ## Target column

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Doing transformations on train and test split
            input_feature_train_arr = preprocessor_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformer_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)



