import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np


sys.path.append("../src")

import configures_manner
import data_preparation

repo = "p971074907"
country = "brl"
subregion1 = "rn"
path = country + ":" + subregion1
feature = "date:newDeaths"
window_size = "7"
begin = "2020-01-01"
end = "2020-04-15"

data_configs = {
    "repo": repo,
    "path": path,
    "feature": feature,
    "begin": begin,
    "end": end,
}

from models import naive_manner
from models import average_manner
import evaluator_manner

evaluator = evaluator_manner.Evaluator(data_configs)
gd = evaluator.grid_search_from_local_configures(save_file=False)

best_model = evaluator.get_best_model(gd)
print(best_model)
