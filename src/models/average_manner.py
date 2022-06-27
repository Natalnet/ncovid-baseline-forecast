import numpy as np
from sklearn.metrics import mean_squared_error
import configures_manner
import json


class ModelAverage:
    def __init__(
        self,
        forecast_average_lenght: int,
        calculation_average_lenght: int = None,
        start_from: str = "first",
        average_type: str = "mean",
    ):
        super().__init__()

        self.forecast_average_lenght = forecast_average_lenght
        self.calculation_average_lenght = calculation_average_lenght
        self.start_from = start_from
        self.average_type = getattr(np, average_type)

    def _start_average_calculation_from_first(self, sequence):
        return [
            [
                self.average_type(
                    sequence[
                        self.start_index_for_begin : self.start_index_for_begin
                        + self.calculation_average_lenght
                    ]
                )
            ]
        ]

    def _start_average_calculation_from_last(self, sequence):
        return [[self.average_type(sequence[-self.calculation_average_lenght :])]]

    def __one_step_average_calculation(self, sequence: np.array) -> np.array:
        pred_average = getattr(
            self, f"_start_average_calculation_from_{self.start_from}"
        )(sequence)

        sequence = np.append(sequence, pred_average, axis=0)
        return sequence

    def _one_input_forward(self, input_sample: np.array) -> np.array:

        self.calculation_average_lenght = (
            self.calculation_average_lenght
            if self.calculation_average_lenght
            else len(input_sample)
        )

        one_step_forwarded = input_sample

        for start_index in range(self.forecast_average_lenght):
            self.start_index_for_begin = start_index
            one_step_forwarded = self.__one_step_average_calculation(one_step_forwarded)

        return one_step_forwarded[-self.forecast_average_lenght :]

    def predicting(self, batch: np.array) -> np.array:
        output = [self._one_input_forward(input) for input in batch]

        return np.array(output)

    def score_calculator(self, ytrue: np.array, ypred: np.array) -> tuple:
        scores = list()
        for i in range(ytrue.shape[1]):
            mse = mean_squared_error(ytrue[:, i], ypred[:, i])
            rmse = np.sqrt(mse)
            scores.append(rmse)

        score_acumullator = 0
        for row in range(ytrue.shape[0]):
            for col in range(ytrue.shape[1]):
                score_acumullator += (ytrue[row, col] - ypred[row, col]) ** 2
        score = np.sqrt(score_acumullator / (ytrue.shape[0] * ytrue.shape[1]))

        return float(score), scores

    def load_instace_model_from_id(self, model_id: str):
        with open(configures_manner.model_path + model_id + ".json") as json_file:
            model_instance_json = json.load(json_file)

        return model_instance_json
