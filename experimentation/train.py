#!/usr/bin/env python
# coding: utf-8

# # Train a Ridge Regression Model on the Diabetes Dataset

# This notebook loads the Diabetes dataset from sklearn, splits the data into training and validation sets, trains a Ridge regression model, validates the model on the validation set, and saves the model.

# In[6]:


from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

def split_data(df):
    X = df.drop('Y', axis=1).values
    y = df['Y'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data

def train_model(data, args):
    reg_model = Ridge(**args)
    reg_model.fit(data["train"]["X"], data["train"]["y"])
    return reg_model

def get_model_metrics(reg_model, data):
    preds = reg_model.predict(data["test"]["X"])
    mse = mean_squared_error(preds, data["test"]["y"])
    metrics = {"mse": mse}

    return metrics

def main():
    sample_data = load_diabetes()

    df = pd.DataFrame(
        data=sample_data.data,
        columns=sample_data.feature_names)
    df['Y'] = sample_data.target

    data = split_data(df)

    # experiment parameters
    args = {
        "alpha": 0.5
    }
    reg = train_model(data, args)

    metrics = get_model_metrics(reg, data)

    model_name = "sklearn_regression_model.pkl"

    joblib.dump(value=reg, filename=model_name)

main()

