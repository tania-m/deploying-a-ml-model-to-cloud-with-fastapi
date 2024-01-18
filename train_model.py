# Script to train machine learning model.

import pandas as pd
import os
import pickle
from data import process_data
from sklearn.model_selection import train_test_split
from data import process_data
from model import train_model, inference, compute_model_metrics, compute_slice_performance

# Add code to load in the data.
data_location = "data/census_clean.csv"
print(f"Loading data from {data_location}")
data = pd.read_csv(data_location)
print(f"Data loaded from {data_location}!")

# Optional enhancement, use K-fold cross validation instead of a train-test split
print("Train-test split")
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

target_label = "salary"
print("Processing training datasets")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=target_label, training=True
)

# Proces the test data with the process_data function
print("Processing test datasets")
# Reuse encoder and lb from training dataset processing
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label=target_label, training=False,
    encoder=encoder, lb=lb
)

# Train and save a model
print("Training model")
model = train_model(X_train, y_train)
print("Model training DONE!")

# Save model
print("Saving model and supporting files")

model_folder_name = "model"
trained_model_name = "model.pkl"
model_full_path = os.path.join(model_folder_name, trained_model_name)
encoder_name = "encoder.pkl"
encoder_full_path = os.path.join(model_folder_name, encoder_name)
lb_name = "lb.pkl"
lb_full_path = os.path.join(model_folder_name, lb_name)
save_mode = "wb"

with open(model_full_path, save_mode) as model_file:
    pickle.dump(model, open(model_full_path, save_mode))
print(f"Model file saved to {model_full_path}")

with open(encoder_full_path, save_mode) as encoder_file:
    pickle.dump(encoder, encoder_file)
print(f"Encoder file saved to {encoder_full_path}")

with open(lb_full_path, save_mode) as lb_file:
    pickle.dump(lb, lb_file)
print(f"Label binarizer file saved to {lb_full_path}")

# Make predictions based on trained model
print("Making test predictions based on trained model")
y_preds = inference(model, X_test)

# Score the model (based on previous predictions)
# Our compute_model_metrics takes tree params, the last one is the f_beta
# beta value, that defaults to 1. So defaults to giving the f1_score
print("Scoring model")
precision, recall, f1_score = compute_model_metrics(y_test, y_preds)
print(f"Model precision: {precision}")
print(f"Model recall: {recall}")
print(f"Model f1_score: {f1_score}")

# Save f1_score next to model
print("Saving f1_score results next to model")
f1_scoring_results_path = os.path.join(model_folder_name, "f1_score.txt")
with open(f1_scoring_results_path, "w") as scoring_file:
    scoring_file.write(str(f1_score))
    print(f"f1_score saved to {f1_scoring_results_path}")
    
# Compute slices performance
print("Computing slices performance")
compute_slice_performance(
    model,
    encoder, 
    lb, 
    cat_features, 
    ["workclass", "occupation", "sex"], 
    data,
    target_label
)