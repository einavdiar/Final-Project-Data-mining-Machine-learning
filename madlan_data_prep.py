# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 21:18:12 2023

@author: einav
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import string

def prepare_data(data):
    # Delete assets without a price
    data.dropna(subset=['price'], inplace=True)
    data['price'] = pd.to_numeric(data['price'], errors='coerce')
    data.dropna(subset=['price'], inplace=True)
    
    # Delete non-numeric values
    columns_list = ['room_number', 'Area']
    data[columns_list] = data[columns_list].astype(str).replace(r'[^0-9.]', '', regex=True)
    data[columns_list] = data[columns_list].apply(pd.to_numeric, errors='coerce')
    
    # Removing punctuation marks from the texts
    columns_to_clean = ['Street', 'city_area', 'description ', 'City']
    punctuation_pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    data[columns_to_clean] = data[columns_to_clean].applymap(lambda x: punctuation_pattern.sub('', str(x)))
    
    # Adding a floor column and a total_floors column
    data[['floor', 'total_floor']] = data['floor_out_of'].str.split(' מתוך ', expand=True)
    data['floor'] = data['floor'].str.replace('קומה ', '')
    data['floor'] = data['floor'].replace('קומת קרקע', 0)
    data['floor'] = data['floor'].replace('3 תוך 19', 19)
    data['floor'] = data['floor'].replace('קומת מרתף', -1)
    data.drop(columns=['floor_out_of'], inplace=True)
    
    # removing spaces and changing 'city' column
    data['City'] = data['City'].str.replace('נהרייה', 'נהריה')
    data['City'] = data['City'].str.replace(' שוהם', 'שוהם')
    
    # Creating a categorical entrance_date column
    data['entranceDate '] = data['entranceDate '].str.lower()
    conditions = [
        data['entranceDate '].str.contains('מיידי', na=False) | (pd.to_datetime(data['entranceDate '], errors='coerce') < pd.Timestamp.now() + pd.DateOffset(months=6)),
        (pd.to_datetime(data['entranceDate '], errors='coerce') >= pd.Timestamp.now() + pd.DateOffset(months=6)) & (pd.to_datetime(data['entranceDate '], errors='coerce') <= pd.Timestamp.now() + pd.DateOffset(months=12)),
        pd.to_datetime(data['entranceDate '], errors='coerce') > pd.Timestamp.now() + pd.DateOffset(years=1),
        data['entranceDate '].str.contains('גמיש', na=False),
        data['entranceDate '].str.contains('לא צויין', na=False)
    ]
    choices = ['less_than_6 months', 'months_6_12', 'above_year', 'flexible', 'not_defined']
    data['entrance_date'] = np.select(conditions, choices, default='not_defined')
    data.drop(columns=['entranceDate '], inplace=True)
    
    # Converting the boolean fields to 0 and 1
    columns_to_convert = ['hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ']
    data[columns_to_convert] = data[columns_to_convert].apply(lambda x: x.astype(bool).astype(int))

    # Filling in missing values
    data['total_floor'] = data['total_floor'].replace(np.nan, 0)
    data['publishedDays '] = data['publishedDays '].replace(np.nan, 0)
    data['num_of_images'] = data['num_of_images'].replace(np.nan, 0)
    data['publishedDays '] = pd.to_numeric(data['publishedDays '], errors='coerce')
    data['total_floor'] = pd.to_numeric(data['total_floor'], errors='coerce')
    data['floor'] = pd.to_numeric(data['floor'], errors='coerce')
    data['Area'] = pd.to_numeric(data['Area'], errors='coerce')
    data.dropna(subset=['Area'], inplace=True)
    data.dropna(subset=['room_number'], inplace=True)
    data.dropna(subset=['room_number'], inplace=True)
   
    # One-hot encoder on the type column
    data['type'].replace({'דירת גן': 'בית', 'קוטג טורי': 'בית', 'פרטי': 'בית', 'קוטג': 'בית', 'דופלקס': 'בית', 'טריפלקס': 'בית', 'דו משפחתי': 'בית','בית פרטי': 'בית', "קוטג'": 'בית', "קוטג' טורי": 'בית', 'נחלה': 'בית', 'מיני פנטהאוז': 'פנטהאוז','דירת גג': 'פנטהאוז', 'בניין': 'דירה', 'מגרש': 'אחר'}, inplace=True)

    one_hot_encoded = pd.get_dummies(data['type'], prefix='type')
    data = pd.concat([data, one_hot_encoded], axis=1)
    data.drop(columns=['type'], inplace=True)
    
    columns = ['room_number', 'Area', 'hasParking ', 'hasStorage ', 'hasBalcony ', 'hasMamad ', 'type_בית', 'price']
    return data[columns]