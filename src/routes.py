from flask import Flask, jsonify, request

app = Flask(__name__)

import json
import evaluator_manner
import configures_manner
import predictor_manner


@app.route(
    "/api/v1/modelType/<modelType>/train/repo/<repo>/path/<path>/",
    methods=["POST"],
)
def train_baseline(modelType, repo, path):

    metadata_to_train = json.loads(request.form.get("metadata"))

    configures_manner.add_variable_to_globals("model_type", modelType)
    configures_manner.add_variable_to_globals("repo", repo)
    configures_manner.add_variable_to_globals("path", path)
    configures_manner.overwrite(metadata_to_train)

    evaluator = evaluator_manner.Evaluator()
    grid_search_response = evaluator.grid_search_from_request_configures(
        save_file=False
    )

    best_model = evaluator.get_best_model(grid_search_response)

    evaluator.save_best_model()
    reponse_json = jsonify(best_model)

    return reponse_json


@app.route(
    "/api/v1/modelCategory/<modelCategory>/predict/modelInstance/<modelInstance>/begin/<begin>/end/<end>/",
    methods=["GET"],
)
def predict_baseline(modelCategory, modelInstance, begin, end):

    predictor = predictor_manner.Predictor(modelCategory)

    predictor.load_instace_model_from_id(modelInstance)
    data_to_predict = predictor.gen_data_to_predict(begin, end)

    yhat = predictor.predict(data_to_predict)
    yhat_json = predictor.predictions_to_weboutput(yhat, begin, end)

    response_json = jsonify(yhat_json)

    return response_json


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
