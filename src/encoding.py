import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

@staticmethod
def encode_categorical_columns(df):
    label_encoders = {}
    try:
        for col in df.select_dtypes(includes=['object','category']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            
    except Exception as e:
        print(f"Error encoding categorical column '{col}': {e}")
    return df, label_encoders

@staticmethod
def scale_numerical_columns(df, cols):
    scaler = StandardScaler()
    try:
        df[cols] = scaler.fit_transform(df[cols])
    except KeyError as e:
        print(f"Missing columns for scaling: {e}")
    except Exception as e:
        print(f"Error scaling numerical columns: {e}")
    return df, scaler