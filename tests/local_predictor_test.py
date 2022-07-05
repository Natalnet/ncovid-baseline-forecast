import sys

sys.path.append("../src")

import predictor_manner

# Predict Test

## URL contents
modelInstance = "8f3c2d12-f7fa-11ec-92cd-db98f636c7a7"
modelCategory = "naive"

begin = "2020-06-01"
end = "2020-07-09"

# Predictor Flow
predictor = predictor_manner.Predictor(modelCategory)

## Load the local instance
predictor.load_instace_model_from_id(modelInstance)

## Gen data to predict
data_to_predict = predictor.gen_data_to_predict(begin, end)

## Predicting
yhat = predictor.predict(data_to_predict)

## Parsing to desired form
yhat_json = predictor.predictions_to_weboutput(yhat, begin, end)

## Printing
print(yhat_json)
