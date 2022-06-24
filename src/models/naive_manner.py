import numpy as np
from sklearn.metrics import mean_squared_error
import configures_manner
import json


class ModelNaive:
    def __init__(
        self,
        forecast_average_lenght: int,
        naive_mode: str = "begin",
    ):
        super().__init__()

        self.forecast_average_lenght = forecast_average_lenght
        self.naive_mode = naive_mode

    def __one_step_from_random(self, sequence: np.array) -> np.array:
        random_input_index = np.random.randint(len(sequence), size=1)
        random_input_pred = sequence[random_input_index]
        sequence = np.append(sequence, random_input_pred, axis=0)
        return sequence

    def _one_input_from_random_forward(self, input_sample: np.array) -> np.array:
        one_step__random_forwarded = input_sample
        for _ in range(self.forecast_average_lenght):
            one_step__random_forwarded = self.__one_step_from_random(
                one_step__random_forwarded
            )

        return one_step__random_forwarded[-self.forecast_average_lenght :]

    def _one_input_from_begin_forward(self, input_sample: np.array) -> np.array:
        one_step_begin_forwarded = np.resize(
            input_sample, (self.forecast_average_lenght, 1)
        )
        return one_step_begin_forwarded

    def _one_input_from_end_forward(self, input_sample: np.array) -> np.array:
        one_step_end_forwarded = np.flip(input_sample)
        one_step_end_forwarded = np.resize(
            one_step_end_forwarded, (self.forecast_average_lenght, 1)
        )
        return one_step_end_forwarded

    def _one_input_forward(self, input_sample: np.array) -> np.array:
        one_step_forwarded = getattr(
            self, f"_one_input_from_{self.naive_mode}_forward"
        )(input_sample)

        return one_step_forwarded

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
