import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class ElectricityData:
    def __init__(  self,
        stateDescription: str,
        sectorName: str,
        month: int,
        price: int,        
        year: int):        

        self.stateDescription = stateDescription

        self.sectorName = sectorName        

        self.month = month

        self.price = price

        self.year = year

        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "stateDescription": [self.stateDescription],
                "sectorName": [self.sectorName],
                "month": [self.month],
                "price": [self.price],
                "year": [self.year],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)