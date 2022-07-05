import sys

sys.path.append("../src")

import configures_manner
import evaluator_manner

# Evaluator test

# POST request infos

## POST contents
### average example
# metadata_to_train = {
#     "schema": "schema",
#     "entity": "baseline",
#     "inputFeatures": "newDeaths",
#     "outputFeatures": "deaths",
#     "begin": "2020-06-01",
#     "end": "2020-07-09",
#     "inputWindowSize": 7,
#     "outputWindowSize": 7,
#     "testSize": 14,
#     "modelConfigs": {
#         "forecast_average_lenght": [7, 14, 21, 28],
#         "calculation_average_lenght": [7, 14, 21],
#         "start_from": ["first", "last"],
#         "average_type": ["mean", "median"],
#     },
# }

### naive example
metadata_to_train = {
    "schema": "schema",
    "entity": "baseline",
    "inputFeatures": "newDeaths",
    "outputFeatures": "deaths",
    "begin": "2020-06-01",
    "end": "2020-07-09",
    "inputWindowSize": 7,
    "outputWindowSize": 7,
    "testSize": 14,
    "modelConfigs": {
        "forecast_average_lenght": [5, 7, 10],
        "naive_mode": ["random", "ordered", "inverted"],
    },
}

## URL contents
repo = "p971074907"
path = "brl:rn"

modelType = "naive"

# Add those variables to configures
configures_manner.add_variable_to_globals("model_type", modelType)
configures_manner.add_variable_to_globals("repo", repo)
configures_manner.add_variable_to_globals("path", path)
configures_manner.overwrite(metadata_to_train)

# Evaluator flow
evaluator = evaluator_manner.Evaluator()

## Searching from requested values
gd = evaluator.grid_search_from_request_configures(save_file=False)

## Find best model
best_model = evaluator.get_best_model(gd)

## Saving
evaluator.save_best_model()

## Printing
print(best_model)
