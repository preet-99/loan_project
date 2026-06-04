import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformer import DataTransformer
from src.components.model_trainer import ModelTrainer

from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    """
    this function is responsible for Ingestion 
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ## read data from csv file
            df = pd.read_csv("notebook\data\loan.csv")
            ## Remove spaces 
            df.columns = df.columns.str.strip()
            df['education'] = df['education'].str.strip()
            df['self_employed'] = df['self_employed'].str.strip()
            
            df.drop(columns=["loan_id"], inplace=True)

            logging.info("Read dataset as dataframe")

            # make artifacts folder
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # do train and test split
            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            # save train set as a csv
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            # save test set as csv
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Ingestion of the data is Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformer()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))