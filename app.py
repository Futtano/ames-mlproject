from flask import Flask, request, render_template, jsonify
from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route('/')
def index():
        return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
        pred_pipe = PredictionPipeline()
        result = pred_pipe.predict(request.get_json())
        return jsonify({'SalePrice' : float(result[0])})

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)