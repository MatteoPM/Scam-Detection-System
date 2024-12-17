# pip install flask pyspark findspark

from flask import Flask, request, jsonify
from pyspark.sql import SparkSession, Row
from pyspark.ml.pipeline import PipelineModel
import requests
from flask_cors import CORS

# Initialize Flask and Spark
app = Flask(__name__)
CORS(app)
spark = SparkSession.builder.appName("ScamDetection").getOrCreate()

# Load the pre-trained pipeline model
pipeline_model = PipelineModel.load("scam_detection_pipeline_model")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    dialogue = data.get("dialogue")
    if not dialogue:
        return jsonify({"error": "Invalid input"}), 400
    print(dialogue)
    # Create a Spark DataFrame for prediction
    input_data = spark.createDataFrame([Row(dialogue=dialogue)])
    prediction = pipeline_model.transform(input_data).select("prediction").collect()[0]["prediction"]
    label_mapping = {0: "Non-Scam", 1: "Scam"}
    result = label_mapping[int(prediction)]

    # Return prediction
    return jsonify({"prediction": result})


if __name__ == "__main__":
    public_ip = requests.get('https://api.ipify.org').text
    print(f"Running Flask app on public IP: {public_ip}")
    app.run(host='127.0.0.1', port=4545)
