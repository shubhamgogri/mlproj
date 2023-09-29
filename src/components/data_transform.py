# feature engineering.
import sys # reqiured for the exceptions handling. 
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler # for categorical and numerical data
from sklearn.impute import SimpleImputer # for missing data
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # path for saving the pkl file.
    preprocessor_ob_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        # actual conversion of the categorical data and numerical data.
        try:
            numerical_feature = ['writing_score', 'reading_score']
            categorical_feature = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # 1. Create Pipeline
            # 2. impute the missing values. 

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical columns standard scaling completed.")

            logging.info("Categorical columns encoding completed.")

            prepocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_feature), 
                    ("categorical_pipeline", categorical_pipeline, categorical_feature) 
                ]
            )

            return prepocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train, test):
        try:
            train_df =pd.read_csv(train)
            test_df = pd.read_csv(test)

            logging.info("Read the data For data Transformation technique")
            target_column = "math_score"

            # get the preprocessor object
            preprocessor_obj = self.get_data_transformer_object()
            # numerical_column = ['writing_score', 'reading_score']
            
            # dependant and independant feature splits for each train and test.
            input_feature_train_df= train_df.drop(columns=[target_column], axis= 1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df= test_df.drop(columns=[target_column], axis= 1)
            target_feature_test_df = test_df[target_column]
            
            logging.info(f"Applying the preprocessing object on training dataframe ans test df.")

            # now transform the data using the preprocessing object. 
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            
            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessor_obj
            )

            logging.info(f"Saved preprocessing object.")
            return(
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)