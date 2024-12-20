# pip install flask pyspark findspark

from flask import Flask, request, jsonify
from pyspark.sql import SparkSession, Row
from pyspark.ml.pipeline import PipelineModel
import requests
from flask_cors import CORS
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import lower, regexp_replace
import sparknlp


# Initialize Flask and Spark
app = Flask(__name__)
CORS(app)
# spark = SparkSession.builder.appName("ScamDetection").getOrCreate()


# Initialize Spark Session
spark = sparknlp.start()

# Load the pre-trained pipeline model
model = PipelineModel.load("random_forest_model")
print("Model loaded successfully.")

# Function to predict on new text


def predict_scam_or_not(text):
    # Create a single row DataFrame
    data = [(text,)]
    df = spark.createDataFrame(data, ["dialogue"])
    # Make prediction
    prediction = model.transform(df)
    result = prediction.select("prediction").collect()[0][0]

    return "Scam" if result == 1 else "Non-Scam"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    dialogue = data.get("dialogue")
    print(dialogue)
    if not dialogue:
        return jsonify({"error": "Invalid input"}), 400
    return jsonify({"prediction": predict_scam_or_not(dialogue)})


if __name__ == "__main__":
    public_ip = requests.get('https://api.ipify.org').text
    print(f"Running Flask app on public IP: {public_ip}")
    app.run(host='127.0.0.1', port=4545)
