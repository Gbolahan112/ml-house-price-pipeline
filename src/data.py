# src/data.py

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data():
    housing = fetch_california_housing()
    
    df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )
    
    df["Price"] = housing.target
    
    return df