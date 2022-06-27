import configures_manner
import json, uuid
import itertools
from models import average_manner
from models import naive_manner
import numpy as np
from data_preparation import DataPreparation
from datetime import date, datetime


class Evaluator:
    def __init__(self, data_configs: dict):
        super().__init__()
        for config, value in data_configs.items():
            configures_manner.add_variable_to_globals("data_" + str(config), value)

    def grid_search_from_request_configures(
        self, save_file: bool = True, results_file_name: str = None
    ) -> dict():
        combinations = self._set_configures_combinations()
        model = configures_manner.model_type
        grid_search_dictionary = getattr(self, f"_{model}_grid_search")(combinations)
        if save_file:
            self.__save_grid_search_json_file(grid_search_dictionary, results_file_name)
        return grid_search_dictionary

    def grid_search_from_local_configures(
        self, save_file: bool = True, results_file_name: str = None
    ) -> dict():
        combinations = self._get_configures_combinations()
        model = configures_manner.model_type
        grid_search_dictionary = getattr(self, f"_{model}_grid_search")(combinations)
        if save_file:
            self.__save_grid_search_json_file(grid_search_dictionary, results_file_name)
        return grid_search_dictionary

    def _get_configures_combinations(self) -> list():
        with open("../doc/configure_grid_search.json") as json_file:
            configure_grid_search_dict = json.load(json_file)
            configures_manner.add_all_configures_to_globals(configure_grid_search_dict)

        configures_variation = list()
        for value in configures_manner.model_configs.values():
            configures_variation.append(value)

        configures_combinations = list(itertools.product(*configures_variation))

        return configures_combinations

    def _set_configures_combinations(self) -> list():
        configures_variation = list()
        for value in configures_manner.model_configs.values():
            configures_variation.append(value)

        configures_combinations = list(itertools.product(*configures_variation))

        return configures_combinations

    def _average_grid_search(self, combinations) -> dict():
        grid_search_results_dict = dict()
        for model_index, combination in enumerate(combinations):
            data_obj = DataPreparation()
            data_obj.get_data(
                configures_manner.data_repo,
                configures_manner.data_path,
                configures_manner.data_feature,
                configures_manner.data_window_size,
                configures_manner.data_begin,
                configures_manner.data_end,
            )
            inputs, targets = data_obj.target_data_gen(combination[0])

            forecast_average_lenght = combination[0]
            calculation_average_lenght = combination[1]
            start_from = combination[2]
            average_type = combination[3]

            model = average_manner.ModelAverage(
                forecast_average_lenght,
                calculation_average_lenght,
                start_from,
                average_type,
            )
            average_prediction = model.predicting(inputs)
            score, scores = model.score_calculator(targets, average_prediction)

            grid_search_results_dict["model_index_" + str(model_index)] = {
                "model_id": str(uuid.uuid1()),
                "baseline_type": "average",
                "params": {
                    "forecast_average_lenght": combination[0],
                    "calculation_average_lenght": combination[1],
                    "start_from": combination[2],
                    "average_type": combination[3],
                },
                "score": score,
                "scores": scores,
            }

        return grid_search_results_dict

    def _naive_grid_search(self, combinations) -> dict():
        grid_search_results_dict = dict()

        for model_index, combination in enumerate(combinations):
            data_obj = DataPreparation()
            data_obj.get_data(
                configures_manner.data_repo,
                configures_manner.data_path,
                configures_manner.data_feature,
                configures_manner.data_window_size,
                configures_manner.data_begin,
                configures_manner.data_end,
            )
            inputs, targets = data_obj.target_data_gen(combination[0])

            forecast_average_lenght = combination[0]
            naive_mode = combination[1]

            model = naive_manner.ModelNaive(forecast_average_lenght, naive_mode)
            average_prediction = model.predicting(inputs)
            score, scores = model.score_calculator(targets, average_prediction)

            grid_search_results_dict["model_index_" + str(model_index)] = {
                "model_id": str(uuid.uuid1()),
                "baseline_type": "naive",
                "params": {
                    "forecast_average_lenght": combination[0],
                    "naive_mode": combination[1],
                },
                "score": score,
                "scores": scores,
            }

        return grid_search_results_dict

    def __save_grid_search_json_file(
        self,
        dictionary_to_save: dict,
        file_name: str = None,
    ) -> None:
        if file_name is None:
            file_name = date.today()

        dt_string = datetime.now().strftime("%H:%M:%S")
        with open(
            configures_manner.model_path
            + str(file_name)
            + "-evaluation-"
            + dt_string
            + ".json",
            "w",
        ) as fp:
            json.dump(dictionary_to_save, fp, indent=4)

    def get_best_model(self, grid_seach_dict: dict) -> tuple:
        scores_and_keys = {
            item: value["score"] for item, value in grid_seach_dict.items()
        }

        model_index_min_score = min(scores_and_keys, key=scores_and_keys.get)
        best_model_instance = grid_seach_dict[model_index_min_score]
        self.best_model_instance = best_model_instance
        return best_model_instance

    def save_best_model(self) -> None:
        with open(
            configures_manner.model_path
            + self.best_model_instance["model_id"]
            + ".json",
            "w",
        ) as fp:
            json.dump(self.best_model_instance, fp, indent=4)
