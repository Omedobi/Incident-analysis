import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class DataHandling:
    
    def ConvertValue(self, df):
        try:
            # Identify categorical columns
            categorical_values = df.select_dtypes(include=['object']).columns
            
            # Replace None with np.nan in categorical columns
            df[categorical_values] = df[categorical_values].applymap(lambda x: np.nan if x is None or x == 'unknown' else x)
            
            print('None and unknown values successfully replaced with np.nan in categorical columns')
            return df
        except Exception as e:
            print(f'Failed to replace None and unknown values: {e}')
            return None

    def FillMissingValues(self, df):
        try:
            # Identify numerical and categorical columns
            categorical_values = df.select_dtypes(include=['object']).columns
            numerical_values = df.select_dtypes(include=['number']).columns
            
            # Replace None with np.nan in categorical columns
            df[categorical_values] = df[categorical_values].applymap(lambda x: np.nan if x is None else x)
            
            # Initialize SimpleImputer for numerical columns and categorical columns
            cat_imputer = SimpleImputer(strategy='most_frequent')
            num_imputer = SimpleImputer(strategy='mean')
            
            # Apply the imputers to the respective columns
            df[categorical_values] = cat_imputer.fit_transform(df[categorical_values])
            df[numerical_values] = num_imputer.fit_transform(df[numerical_values])
            
            print('Missing values successfully handled')
            return df
        except Exception as e:
            print(f'Failed to fill the missing values: {e}')
            return None