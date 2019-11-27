#!/usr/bin/env python3

# Library imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from math import sqrt
import pickle as pkl
import os
import time
import argparse
from datetime import datetime
import calendar

# Project imports
from features import * # for proc_data()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Whether to train a new model', default=False, action='store_true')
    parser.add_argument('--export', help='Whether to predict the test set', default=False, action='store_true')
    parser.add_argument('--no_validation_set', help='Specify this to not use a validation set (for exporting)', default=True, action='store_false')
    parser.add_argument('--small_train_set', help='Use a small training set for quick evaluation', default=False, action='store_true')
    parser.add_argument('--small_model', help='Use a small model for quick evaluation', default=False, action='store_true')
    args = parser.parse_args()

    if (not args.train) and (not args.export):
        parser.print_help()
    else:
        if args.train:
            print(' -> Training')
            train(use_test_set=args.no_validation_set, small_training=args.small_train_set, small_model=args.small_model)
        if args.export:
            print(' -> Exporting')
            export(filename=str(calendar.timegm(datetime.utcnow().utctimetuple())) + '.csv')

def train(use_test_set=True, small_training=False, small_model=False):
    os.environ['DD_EXPORT_PROJECT'] = 'False'
    start = time.time()
    try:
        print(' -> Loading (%fs)' % (time.time() - start))
        df = pd.read_csv('tcd-ml-1920-group-income-train.csv', low_memory=False)

        if small_training == True:
            print(' -> Using a small training set (30%)')
            df = df.sample(frac=0.3, random_state=605865101)

        print(' -> Cleaning (%fs)' % (time.time() - start))
        df_out = proc_data(df)
        df_out.info()

        X_train, X_test, y_train, y_test,  = split_scale_data(df_out, use_test_set=use_test_set)

        if small_model == True:
            print(' -> Using a small model')
            func = GradientBoostingRegressor(
                random_state=726463305,
                n_estimators=50,
            )
        else:
            print(' -> Using a large model, this will take a long time to train')
            func = GradientBoostingRegressor(
                random_state=726463305,
                n_estimators=1000,
                max_depth=8,
            )
        print(' -> Training (%fs)' % (time.time() - start))
        func.fit(X_train, y_train)
        with open(os.path.join('pickle', 'model.pkl'), mode='wb') as file:
            pkl.dump(func, file)
        print(' -> Calculating score (%fs)' % (time.time() - start))
        print('Score = %f %%' % (func.score(X_test, y_test) * 100))
        print('RMSE  = %s' % (sqrt(mean_squared_error(np.exp(func.predict(X_test)), np.exp(y_test)))))
        print('MAE   = %f'% (mean_absolute_error(np.exp(y_test), np.exp(func.predict(X_test)))))
    finally:
        print(' -> Finished (%fs)' % (time.time() - start))

def export(filename='output.csv'):
    os.environ['DD_EXPORT_PROJECT'] = 'True'
    start = time.time()
    try:
        print(' -> Loading (%fs)' % (time.time() - start))
        df = pd.read_csv('tcd-ml-1920-group-income-test.csv', low_memory=False)
        
        print(' -> Cleaning (%fs)' % (time.time() - start))
        df_out = proc_data(df)
        df_out.info()

        X = split_scale_data(df_out)

        print(' -> Predicting (%fs)' % (time.time() - start))

        with open(os.path.join('pickle', 'model.pkl'), mode='rb') as file:
            func = pkl.load(file)
        
        y = func.predict(X)
        output = []
        for i in range(len(y)):
            output.append([i + 1, np.exp(y[i]) + 1])

        np.savetxt(filename, output, delimiter=',', fmt='%d,%.2f', header='Instance,Total Yearly Income [EUR]', comments='')
    finally:
        print(' -> Finished (%fs)' % (time.time() - start))

def proc_data(df):
    return pd.concat([
        # proc_Instance(df.copy()),
        proc_Year_of_Record(df.copy()),
        proc_Housing_Situation(df.copy()),
        proc_Crime_Level_in_the_City_of_Employement(df.copy()),
        proc_Work_Experience_in_Current_Job_years(df.copy()),
        proc_Satisfation_with_employer(df.copy()),
        proc_Gender(df.copy()),
        proc_Age(df.copy()),
        proc_Country(df.copy()),
        proc_Size_of_City(df.copy()),
        proc_Profession(df.copy()),
        proc_University_Degree(df.copy()),
        proc_Wears_Glasses(df.copy()),
        proc_Hair_Color(df.copy()),
        proc_Body_Height_cm(df.copy()),
        proc_Yearly_Income_in_addition_to_Salary(df.copy()),
        proc_Total_Yearly_Income_EUR(df.copy()),
        get_scaled_features(df.copy()),
    ],axis=1,)

def split_scale_data(df, use_test_set=True):
    if os.environ['DD_EXPORT_PROJECT'] == 'False':
        if use_test_set == True:
            df = df.fillna(0)
            X = df.drop(columns=['Total Yearly Income [EUR]']).to_numpy()
            y = df['Total Yearly Income [EUR]']#.to_numpy()
            y = np.log(y + 1)
            return train_test_split(X, y, test_size=0.2, random_state=483304557)
        else:
            print(' -> Not using a validation set')
            df = df.fillna(0)
            X = df.drop(columns=['Total Yearly Income [EUR]']).to_numpy()
            y = df['Total Yearly Income [EUR]']#.to_numpy()
            y = np.log(y + 1)
            return X, X, y, y
    elif os.environ['DD_EXPORT_PROJECT'] == 'True':
        X = df.drop(columns=['Total Yearly Income [EUR]']).to_numpy()
        return X
    else:
        raise RuntimeError('OS Enviornment variable \'DD_EXPORT_PROJECT\' not set')

if __name__ == "__main__":
    main()