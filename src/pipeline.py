import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: User Inputs for Paths and Settings
def get_user_inputs():
    """
    Prompt user to input necessary file paths and settings.
    """
    print("Welcome! Please provide the necessary file paths and settings.")
    dataset_path = input("Enter the path to your dataset (CSV file): ").strip()
    model_type = input("Enter the model type you want to use (xgboost/lightgbm): ").strip().lower()
    target_column = input("Enter the name of the target column: ").strip()
    model_save_path = input("Enter the path to save/load the model (e.g., 'model.pkl'): ").strip()
    encoders_save_path = input("Enter the path to save/load the encoders (e.g., 'encoders.pkl'): ").strip()
    return dataset_path, model_type, target_column, model_save_path, encoders_save_path

# Step 2: Encoding Features
def encode_features(df, target_column):
    """
    Encodes categorical features using LabelEncoder.
    """
    encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col != target_column:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder
    return df, encoders

# Step 3: Training the Model
def train_model(dataset_path, model_type, target_column, model_save_path, encoders_save_path):
    """
    Train the selected model and save it along with the encoders.
    """
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Encode features
    df, encoders = encode_features(df, target_column)

    # Encode target column
    target_encoder = LabelEncoder()
    df[target_column] = target_encoder.fit_transform(df[target_column])
    encoders[target_column] = target_encoder

    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Select model type
    if model_type == 'xgboost':
        model = XGBClassifier(random_state=42)
    elif model_type == 'lightgbm':
        model = LGBMClassifier(random_state=42)
    else:
        raise ValueError("Invalid model type. Choose either 'xgboost' or 'lightgbm'.")

    # Train the model
    model.fit(X, y)

    # Save the model and encoders
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    with open(encoders_save_path, 'wb') as f:
        pickle.dump(encoders, f)

    print("Model and encoders saved successfully!")

# Step 4: Prediction with Preprocessing
def preprocess_and_predict(data_path, model_path, encoders_path):
    """
    Load model and encoders, preprocess raw data, and make predictions.
    """
    # Load raw data
    raw_data = pd.read_csv(data_path)

    # Load encoders and model
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Encode raw data
    for col, encoder in encoders.items():
        if col in raw_data.columns:
            raw_data[col] = encoder.transform(raw_data[col])

    # Predict
    predictions = model.predict(raw_data)

    # Decode predictions
    target_encoder = encoders[list(encoders.keys())[-1]] 
    decoded_predictions = target_encoder.inverse_transform(predictions)
    return decoded_predictions

# Step 5: Main Script Execution
if __name__ == "__main__":
    dataset_path, model_type, target_column, model_save_path, encoders_save_path = get_user_inputs()

    # Train the model
    train_model(dataset_path, model_type, target_column, model_save_path, encoders_save_path)

    # Predict on unseen data
    unseen_data_path = input("Enter the path to the unseen data (CSV file) for prediction: ").strip()
    predictions = preprocess_and_predict(unseen_data_path, model_save_path, encoders_save_path)

    # Display predictions
    print(f"Predictions: \n {predictions}")
