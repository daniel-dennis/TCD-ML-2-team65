"""
This file contains a function that contains a function for making a new feature,
it takes the entire dataframe as an input, and returns a dataframe containing
only the new feature(s)
"""

# Library imports
import pandas as pd
import numpy as np

# Project imports
from feature_helpers import fit_to_value, impute_values, one_hot_encode, label_encode_values

def proc_Instance(df):
    return df['Instance']
    
def proc_Year_of_Record(df):
    return impute_values(df, 'Year of Record')
    
def proc_Housing_Situation(df):
    # df['Housing Situation'] = df['Housing Situation'].fillna('NA')
    return fit_to_value(df, 'Housing Situation')
    
def proc_Crime_Level_in_the_City_of_Employement(df):
    df['Crime Level in the City of Employement'] = pd.to_numeric(df['Crime Level in the City of Employement'], errors='coerce')
    df = pd.DataFrame(df['Crime Level in the City of Employement'].fillna(0))
    return df
    
def proc_Work_Experience_in_Current_Job_years(df):
    df['Work Experience in Current Job [years]'] = pd.to_numeric(df['Work Experience in Current Job [years]'], errors='coerce')
    df = pd.DataFrame(df['Work Experience in Current Job [years]'].fillna(0))
    return df
    
def proc_Satisfation_with_employer(df):
    return fit_to_value(df, 'Satisfation with employer')
    
def proc_Gender(df):
    return fit_to_value(df, 'Gender')
    
def proc_Age(df):
    return df['Age']
    
def proc_Country(df):
    return fit_to_value(df, 'Country')
    
def proc_Size_of_City(df):
    return df['Size of City']
    
def proc_Profession(df):
    df['Profession'] = df['Profession'].str.split().str[0]
    return fit_to_value(df, 'Profession')
    
def proc_University_Degree(df):
    df = pd.DataFrame(df['University Degree'].fillna('0'))
    return one_hot_encode(df, 'University Degree')
    
def proc_Wears_Glasses(df):
    return df['Wears Glasses']
    
def proc_Hair_Color(df):
    df = pd.DataFrame(df['Hair Color'].fillna('0'))
    return one_hot_encode(df, 'Hair Color')
    
def proc_Body_Height_cm(df):
    return df['Body Height [cm]']
    
def proc_Yearly_Income_in_addition_to_Salary(df):
    array = df['Yearly Income in addition to Salary (e.g. Rental Income)'].to_numpy()
    for i in range(len(array)):
        array[i] = float(array[i][:-4])

    return pd.DataFrame(array, columns=['Yearly Income in addition to Salary (e.g. Rental Income)'], dtype=float)
    
def proc_Total_Yearly_Income_EUR(df):
    # array = df['Total Yearly Income [EUR]'].to_numpy()
    # array = np.log(array)
    # df = pd.DataFrame(data=array, columns=['Total Yearly Income [EUR]'])
    return df['Total Yearly Income [EUR]']

def get_scaled_features(df, scale=2):
    df = pd.concat([
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
        # proc_University_Degree(df.copy()),
        proc_Wears_Glasses(df.copy()),
        # proc_Hair_Color(df.copy()),
        proc_Body_Height_cm(df.copy()),
        proc_Yearly_Income_in_addition_to_Salary(df.copy()),
    ],axis=1,)

    df_scaled = pd.DataFrame()
    for col in list(df):
        try:
            d = df[col].to_numpy()
            d = np.power(d, scale)
            df_scaled = pd.concat([
                df_scaled,
                pd.DataFrame(data=d, columns=['scaled_' + str(scale) + '_' + col])
            ], axis=1,)
        except TypeError:
            print('Col: %s' % (col))
            raise


    return df_scaled
