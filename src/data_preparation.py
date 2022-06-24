import pandas as pd
import numpy as np
import datetime


class DataPreparation:
    def __init__(self):
        super().__init__()

    def get_data(self, repo, path, feature, window_size, begin, end):
        if begin == None:
            begin = "2020-01-01"
        if end == None:
            end = "2050-01-01"

        self.window_size = int(window_size)

        file_name = "".join(
            f"http://ncovid.natalnet.br/datamanager/"
            f"repo/{repo}/"
            f"path/{path}/"
            f"features/{feature}/"
            f"window-size/{window_size}/"
            f"begin/{begin}/"
            f"end/{end}/as-csv"
        )
        df = pd.read_csv(file_name, parse_dates=["date"], index_col="date")
        # TODO: solve this for multivariate case
        for x in feature.split(":")[1:]:
            data = df[x].values
        # store first and last day available
        self.begin_raw = df.index[0]
        self.end_raw = df.index[-1]
        self.data = data
        self.instance_region = path
        return data

    def get_data_to_web_request(self, repo, path, feature, window_size, begin, end):
        if begin == None:
            begin = "2020-01-01"
        if end == None:
            end = "2050-01-01"

        self.window_size = int(window_size)

        begin_delay = datetime.datetime.strptime(
            begin, "%Y-%m-%d"
        ) - datetime.timedelta(days=self.window_size)
        new_begin = datetime.datetime.strftime(begin_delay, "%Y-%m-%d")

        period_date_request = pd.date_range(new_begin, end)
        date_gap_to_assert_length = len(period_date_request) % self.window_size
        new_end_datetime = datetime.datetime.strptime(
            end, "%Y-%m-%d"
        ) + datetime.timedelta(days=date_gap_to_assert_length)

        new_end = datetime.datetime.strftime(new_end_datetime, "%Y-%m-%d")

        file_name = "".join(
            f"http://ncovid.natalnet.br/datamanager/"
            f"repo/{repo}/"
            f"path/{path}/"
            f"features/{feature}/"
            f"window-size/{window_size}/"
            f"begin/{new_begin}/"
            f"end/{new_end}/as-csv"
        )
        df = pd.read_csv(file_name, parse_dates=["date"], index_col="date")
        # TODO: solve this for multivariate case
        for x in feature.split(":")[1:]:
            data = df[x].values
        # store first and last day available
        self.begin_raw = df.index[0]
        self.end_raw = df.index[-1]
        self.data = data
        self.instance_region = path
        return data

    def windowing_data(self):
        start_data_index = len(self.data) % self.window_size

        lenght_asserted_data = self.data[start_data_index:]
        lenght_splited_data = int(len(lenght_asserted_data) / self.window_size)
        windowed_data = lenght_asserted_data.reshape(
            lenght_splited_data, self.window_size, 1
        )
        self.windowed_data = windowed_data
        return windowed_data

    def target_data_gen(self, target_len):
        data_inputs = self.data[:-target_len]

        start_data_index = len(data_inputs) % self.window_size
        lenght_asserted_data = data_inputs[start_data_index:]
        lenght_splited_data = int(len(lenght_asserted_data) / self.window_size)
        windowed_data = lenght_asserted_data.reshape(
            lenght_splited_data, self.window_size, 1
        )
        x = windowed_data

        # gen targets
        targets = list()
        for index in range(
            start_data_index + self.window_size, len(data_inputs) + 1, self.window_size
        ):
            start_target_index = index
            end_target_index = index + target_len
            target = self.data[start_target_index:end_target_index]
            targets.append(target)

        y = np.array(targets).reshape(len(targets), target_len, 1)

        self.data_inputs = x
        self.data_targets = y

        return x, y
