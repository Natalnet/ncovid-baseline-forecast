import sys

sys.path.append("../src")


import evaluator_manner

# Evaluator test

## Evaluator flow

evaluator = evaluator_manner.Evaluator()

## Searching from a local file values
gd = evaluator.grid_search_from_local_configures(save_file=False)

## Find best model
best_model = evaluator.get_best_model(gd)

## Saving
evaluator.save_best_model()

## Printing
print(best_model)
