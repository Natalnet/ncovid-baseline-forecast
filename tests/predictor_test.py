import sys

sys.path.append("../src")

import predictor_manner

# Predict Test

# GET request infos

## URL contents
modelInstance = "4b64531a-077e-11ed-90b4-d306882a1848"

model_insatnce_id = modelInstance

begin = "2020-06-01"
end = "2020-07-09"

# Predictor Flow
predictor = predictor_manner.Predictor()

## Load the instance
predictor.load_instace_model_from_id(model_insatnce_id)

## Gen data to predict
data_to_predict = predictor.gen_data_to_predict(begin, end)

## Predicting
yhat = predictor.predict(data_to_predict)

## Parsing to desired form
yhat_json = predictor.predictions_to_weboutput(yhat, begin, end)

## Printing
print(yhat_json)
