import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transform import DataTransformation, DataTransformationConfig

# this class consists of the file paths of all the csv data files.  
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts", "train.csv")
    test_data_path: str= os.path.join("artifacts", "test.csv")
    raw_data_path: str= os.path.join("artifacts", "data.csv")
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # read the datset
            df =pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the Dataset as df')
            # made a directory for artifacts i.e. the dataset csv files, only if the directory doesn't exists. 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok =True)
            # saving the raw file with complete data. 
            df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True)
            logging.info("Train test split intiated")
            # train test split. 
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state=42)
            # saving the training part of the split.
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header= True)
            # saving the csv for the testing part of the data. (20%)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header= True)
            logging.info("Ingestion is Completed.")
            # returning the paths of train and test data files.
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)


if __name__ =="__main__":
    data_ingestion_obj =DataIngestion()
    train_data, test_data = data_ingestion_obj.initiate_data_ingestion()
    
    data_transformation_obj = DataTransformation()
    data_transformation_obj.initiate_data_transformation(train_data, test_data)
    