"""
This file is for methods that are common among multiple features in features.py
"""

# Library imports
import pandas as pd
import numpy as np
import pickle as pkl
import os
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer

def fit_to_value(df, column, income_col='Total Yearly Income [EUR]'):
    """
    Calculates the average income for each category in a column of a dataframe

    ## Parameters
    data: a pandas.DataFrame containing the data

    column: an str containing the column to be processed
    ## Returns
    The a single row pandas.DataFrame containing the processed data
    """
    if os.environ['DD_EXPORT_PROJECT'] == 'False':
        values = pd.Series.to_dict(df[column])
        incomes = pd.Series.to_dict(df[income_col])
        assert(len(values) == len(incomes))
        fitted = {}
        for key in values:
            values_key = values[key]
            income_key = incomes[key]
            try:
                fitted[values_key].append(income_key)
            except KeyError:
                fitted[values_key] = []
                fitted[values_key].append(income_key)
        
        for key in fitted:
            fitted[key] = sum(fitted[key]) / len(fitted[key])
        with open(os.path.join('pickle', column + '_fit_to_value.pkl'), mode='wb') as file:
            pkl.dump(fitted, file)
    elif os.environ['DD_EXPORT_PROJECT'] == 'True':
        with open(os.path.join('pickle', column + '_fit_to_value.pkl'), mode='rb') as file:
            fitted = pkl.load(file)
        values = pd.Series.to_dict(df[column])
        new_fitted = {}
        for key in values:
            new_fitted[values[key]] = 68509.733397
        for key in new_fitted:
            if key not in fitted:
                fitted[key] = new_fitted[key]
    else:
        raise RuntimeError('OS Enviornment variable \'DD_EXPORT_PROJECT\' not set')
    newvals = {column: fitted}
    df = df.replace(to_replace=newvals)
    return df[column]

def impute_values(df, column, strategy='median'):
    """
    Uses scikit-learn's imputer and pickles the instance

    ## Parameters
    data: a pandas.DataFrame containing the data

    column: an str containing the column to be processed

    strategy: can be - 'mean', 'median', 'most_frequent', 'constant', see the 
        sklearn docs
    ## Returns
    The a single row pandas.DataFrame containing the processed data
    """
    return __fit_transform__(df, column, SimpleImputer(strategy=strategy), 'impute_values')

def label_encode_values(df, column):
    """
    Uses scikit-learn's label encoder to replace and pickles the instance
        strings with int values

    ## Parameters
    data: a pandas.DataFrame containing the data

    column: an str containing the column to be processed
    ## Returns
    The a single row pandas.DataFrame containing the processed data
    """
    return __fit_transform__(df, column, LabelEncoder(), 'label_encode_values')

def one_hot_encode(df, column):
    """
    Uses scikit-learn's label encoder to replace and pickles the instance
        strings with int values

    ## Parameters
    data: a pandas.DataFrame containing the data

    column: an str containing the column to be processed
    ## Returns
    The a single row pandas.DataFrame containing the processed data
    """
    if os.environ['DD_EXPORT_PROJECT'] == 'False':
        degree_encoder = LabelBinarizer()
        degree_encoder.fit(df[column])
        with open(os.path.join('pickle', 'one_hot_encode_' + column.replace(' ', '_') + '.pkl'), mode='wb') as file:
            pkl.dump(degree_encoder, file)
    elif os.environ['DD_EXPORT_PROJECT'] == 'True':
        with open(os.path.join('pickle', 'one_hot_encode_' + column.replace(' ', '_') + '.pkl'), mode='rb') as file:
            degree_encoder = pkl.load(file)
    else:
        raise RuntimeError('OS Enviornment variable \'DD_EXPORT_PROJECT\' not set')
    transformed = degree_encoder.transform(df[column])
    ohe_df = pd.DataFrame(transformed)
    return pd.concat([df, ohe_df], axis=1).drop([column], axis=1)
    # return __fit_transform__(df, column, OneHotEncoder(), 'one_hot_encode')   

def __fit_transform__(df, column, skl_instance, pkl_str):
    array = df[column].to_numpy().reshape(-1, 1)
    if os.environ['DD_EXPORT_PROJECT'] == 'False':
        skl_instance.fit(array)
        with open(os.path.join('pickle', column + pkl_str + '.pkl'), mode='wb') as file:
            pkl.dump(skl_instance, file)
    elif os.environ['DD_EXPORT_PROJECT'] == 'True':
        with open(os.path.join('pickle', column + pkl_str + '.pkl'), mode='rb') as file:
            skl_instance = pkl.load(file)
    else:
        raise RuntimeError('OS Enviornment variable \'DD_EXPORT_PROJECT\' not set')
    array = skl_instance.transform(array)
    return pd.DataFrame(array, columns=[column])
