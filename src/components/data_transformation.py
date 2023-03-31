import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os

# ColumnTransformer is used for creating pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformconfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransform:
    def __init__(self):
        self.data_transform_config= DataTransformconfig()
    
    def get_data_transform_obj(self):
        # To creating pkl files which is responsible for data transformation
        try:
            numerical_cols = ['writing_score', 'reading_score']
            categorical_cols = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy= 'median')),
                ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('onehotencoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Categorical encoding and Standard scaling is completed')

            # To combine categoical pipeline and numerical pipeline together
            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
                ]
            ) 

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    # Starting Data Transformation
    def initiate_data_transform(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            logging.info("Read train and test data")

            preprocessing_obj = self.get_data_transform_obj()

            target_col = 'math_score'
            numerical_cols = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_col], axis= 1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col], axis= 1)
            target_feature_test_df = test_df[target_col]

            logging.info("Applying preprocessing on train dataframe and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
                ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f'Saved preprocessing object')

            save_object(
                file_path = self.data_transform_config.preprocessor_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
            
