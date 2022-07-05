from numpy import average
from models import average_manner, naive_manner
import configures_manner
import json
from data_preparation import DataPreparation
import pandas as pd
import datetime


class Predictor:
    def __init__(self, model_predictor_baseline_type: str):
        super().__init__()
        self.model_predictor_baseline_type = model_predictor_baseline_type

    def generate_average_instance(self):
        configures_manner.add_all_configures_to_globals(self.model_instance_json)
        model_instance = average_manner.ModelAverage(
            configures_manner.forecast_average_lenght,
            configures_manner.calculation_average_lenght,
            configures_manner.start_from,
            configures_manner.average_type,
        )
        return model_instance

    def generate_naive_instance(self):
        configures_manner.add_all_configures_to_globals(self.model_instance_json)
        model_instance = naive_manner.ModelNaive(
            configures_manner.forecast_average_lenght, configures_manner.naive_mode
        )
        return model_instance

    def load_instace_model_from_id(self, model_id: str):
        with open(configures_manner.model_path + model_id + ".json") as json_file:
            self.model_instance_json = json.load(json_file)
        self.model_instance = getattr(
            self, f"generate_{self.model_predictor_baseline_type}_instance"
        )()

    def gen_data_to_predict(self, begin: str, end: str):
        configures_manner.add_variable_to_globals("begin", begin)
        configures_manner.add_variable_to_globals("end", end)
        data_obj = DataPreparation()
        data_obj.get_data(
            configures_manner.repo,
            configures_manner.path,
            configures_manner.inputFeatures,
            configures_manner.inputWindowSize,
            configures_manner.begin,
            configures_manner.end,
        )
        return data_obj.windowing_data()

    def predict(self, data_to_predict):
        yhat = self.model_instance.predicting(data_to_predict)
        return yhat.reshape(-1)

    def predictions_to_weboutput(self, yhat, begin, end):
        period = pd.date_range(begin, end)
        returned_dictionary = list()
        for date, value in zip(period, yhat):
            returned_dictionary.append(
                {
                    "date": datetime.datetime.strftime(date, "%Y-%m-%d"),
                    "prediction": str(value),
                }
            )
        return returned_dictionary
