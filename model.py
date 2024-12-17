# Required Libraries
# !pip install pyspark
# !pip install findspark
# !pip install nltk
# pip install huggingface_hub

import re
import nltk
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import Row
from pyspark.ml.feature import NGram

# Initialize Spark Session
spark = SparkSession.builder.appName("ScamDetection").getOrCreate()
print(spark.version)
# Load datasets using pandas and convert to Spark DataFrame
splits = {'train': 'scam-dialogue_train.csv', 'test': 'scam-dialogue_test.csv'}
train_pd = pd.read_csv("hf://datasets/BothBosu/scam-dialogue/" + splits["train"])
test_pd = pd.read_csv("hf://datasets/BothBosu/scam-dialogue/" + splits["test"])

# Drop the "type" column and convert pandas DataFrame to Spark DataFrame
train_df = spark.createDataFrame(train_pd).drop("type")
test_df = spark.createDataFrame(test_pd).drop("type")

# Text preprocessing pipeline
tokenizer = Tokenizer(inputCol="dialogue", outputCol="words")
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
ngram = NGram(inputCol="filtered_words", outputCol="ngrams")
count_vectorizer = CountVectorizer(inputCol="ngrams", outputCol="raw_features")
idf = IDF(inputCol="raw_features", outputCol="features")
label_indexer = StringIndexer(inputCol="label", outputCol="indexed_label")

# Random Forest Classifier
rf_classifier = RandomForestClassifier(labelCol="indexed_label", featuresCol="features", numTrees=100)

# Build the pipeline
pipeline = Pipeline(stages=[
    tokenizer,
    stopwords_remover,
    ngram,
    count_vectorizer,
    idf,
    label_indexer,
    rf_classifier
])

# Train-test split
train_data, valid_data = train_df.randomSplit([0.8, 0.2], seed=42)

# Train the model
pipeline_model = pipeline.fit(train_data)

# Evaluate the model
predictions = pipeline_model.transform(valid_data)
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexed_label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Validation Accuracy: {accuracy:.2f}")

# Classification Metrics
def compute_classification_metrics(predictions):
    evaluator_precision = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="weightedRecall")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="f1")

    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    confusion_matrix = predictions.groupBy("indexed_label", "prediction").count().orderBy("indexed_label", "prediction")

    return precision, recall, f1, confusion_matrix

precision, recall, f1, confusion_matrix = compute_classification_metrics(predictions)
print("\nClassification Report:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nConfusion Matrix:")
confusion_matrix.show()

# Test the model on test data
test_predictions = pipeline_model.transform(test_df)
test_accuracy = evaluator.evaluate(test_predictions)
print(f"Test Accuracy: {test_accuracy:.2f}")

# User-friendly prediction function
def predict_scam_or_not(transcript):
    print("\nInput Transcript:\n", transcript)
    input_data = spark.createDataFrame([Row(dialogue=transcript)])
    prediction_df = pipeline_model.transform(input_data)
    prediction = prediction_df.select("prediction").collect()[0]["prediction"]
    label_mapping = {0: 'Non-Scam', 1: 'Scam'}
    return label_mapping[int(prediction)]

# Example transcripts for testing
test_idx = 600
transcripts = [
    # Scam transcript
    """
    caller: Hello, this is Professor Armin from the University of Calgary. We've detected suspicious activity related to your course 550 project.
    receiver: Suspicious activity? What do you mean?
    caller: It appears that unauthorized access from North Korea has been detected. We need your student ID and password to secure your account immediately.
    receiver: That sounds alarming. Shouldn't the IT department handle this?
    caller: I'm authorized to assist. Time is critical to prevent data loss. Please provide your credentials now.
    """,
    # Non-Scam transcript
    """
    receiver: Hi Professor Armin, I wanted to discuss the upcoming demo for course 550.
    professor: Of course! I'm excited to see what you and your team have developed.
    receiver: Students have been working hard, and we're ready to present to you, the TAs, and the government of China representatives.
    professor: Excellent. Let's schedule the presentation in Calgary next week.
    """,
    # Scam transcript
    """
    caller: Good afternoon, this is Armin from the University of Calgary's security team. We've noticed irregular activity on your course 550 files.
    receiver: Really? What kind of irregular activity?
    caller: Hackers from North Korea are attempting to steal your work. To prevent this, we need your login details immediately.
    receiver: I'm not comfortable sharing my password over the phone.
    caller: If we don't act now, your project could be compromised, affecting your grade.
    """,
    # Non-Scam transcript
    """
    receiver: Hello Professor Armin, we're ready to showcase our course 550 project.
    professor: That's wonderful! I'm looking forward to your demo.
    receiver: We'll be presenting to you, the TAs, and officials from the government of China in Calgary.
    professo
    """,
    """
    caller: Hi, is this 555-1234? 
    receiver: No, this is 555-5678. You must have dialed the wrong number. 
    caller: Oh, I'm so sorry. I was trying to reach my friend John. 
    receiver: I don't know anyone by that name. You might want to check your phone book again. 
    caller: Yeah, I'll do that. Sorry again for disturbing you. 
    receiver: It's okay, mistakes happen. Have a good day.
    """
    ,
    train_pd.iloc[test_idx]["dialogue"]
]


for idx, transcript in enumerate(transcripts):
    result = predict_scam_or_not(transcript)
    print(f"Transcript {idx + 1} Prediction: {result}")

print("label for text_id: {}".format(train_pd.iloc[test_idx]["label"]))
# Save the trained pipeline model
pipeline_model.write().overwrite().save("scam_detection_pipeline_model")

