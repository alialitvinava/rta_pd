import joblib
from flask import Flask
from flask_restful import Api
from flask import request, jsonify

app = Flask(__name__)
api = Api(app)
@app.route('/')
@app.route('/index')
def home():
    return "aplikacja ze srodowiskiem produkcyjnym API"

@app.route('/api/predict_perceptron', methods=['GET'])
def predykcja():
    sepal_length = float(request.args.get('sl'))
    sepal_width = float(request.args.get('sw'))
    petal_length = float(request.args.get('pl'))
    petal_width = float(request.args.get('pw'))
    dane = [sepal_length, sepal_width, petal_length, petal_width]
    model = joblib.load('model.sav')
    pred = int(model.predict([dane]))
    return jsonify(features = dane, predicted_class = pred)



if __name__ == '__main__':
    app.run()