from flask import Flask, jsonify, request

app = Flask(__name__)

import json
import evaluator_manner
import configures_manner
import predictor_manner


@app.route(
    "/api/v1/baseline/model_type/<modelType>/train/repo/<repo>/path/<path>/feature/<feature>/begin/<begin>/end/<end>/",
    methods=["POST"],
)
# modelTypes: exp_smoothing, arima, sarima
def train_baseline(modelType, repo, path, feature, begin, end):
    # get the metadata
    metadata_to_train = json.loads(request.form.get("metadata"))
    configures_manner.overwrite(metadata_to_train)
    data_configs = {
        "repo": repo,
        "path": path,
        "feature": feature,
        "begin": begin,
        "end": end,
    }
    evaluator = evaluator_manner.Evaluator(data_configs)
    gd = evaluator.grid_search_from_request_configures(save_file=False)

    best_model = evaluator.get_best_model(gd)

    evaluator.save_best_model()

    return jsonify(best_model)


@app.route(
    "/api/v1/baseline/predict/model-instance/<modelInstance>/begin/<begin>/end/<end>/",
    methods=["POST"],
)
def predict_baseline(modelInstance, begin, end):

    metadata_instance = json.loads(request.form.get("metadata"))

    model_id = metadata_instance["model_id"]
    model_type = modelInstance

    predictor = predictor_manner.Predictor(model_type)

    predictor.load_instace_model_from_id(model_id)
    data_to_predict = predictor.gen_data_to_predict(begin, end)

    yhat = predictor.predict(data_to_predict)
    yhat_json = predictor.predictions_to_weboutput(yhat, begin, end)

    response_json = jsonify(yhat_json)

    return response_json


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
