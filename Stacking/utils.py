import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import datetime
import re

class Transformation:

    @staticmethod
    def preprocess_date(df: pd.DataFrame) -> pd.DataFrame:    
        df['Dates'] = pd.to_datetime(df['Dates'])
        df['Year'] = df['Dates'].dt.year
        df['Month'] = df['Dates'].dt.month
        df['Day'] = df['Dates'].dt.day
        df['Hour'] = df['Dates'].dt.hour
        
        return df

    @staticmethod
    def season_of_date(date: datetime.date) -> str:
        year = str(date.year)
        seasons = {'spring': pd.date_range(start='21/03/'+year, end='20/06/'+year),
                'summer': pd.date_range(start='21/06/'+year, end='22/09/'+year),
                'autumn': pd.date_range(start='23/09/'+year, end='20/12/'+year)}
        if date in seasons['spring']:
            return 'spring'
        elif date in seasons['summer']:
            return 'summer'
        elif date in seasons['autumn']:
            return 'autumn'
        else:
            return 'winter'

    @staticmethod   
    def day_period(df: pd.DataFrame) -> pd.DataFrame:    
        df['Day_period'] = df['Hour'].apply(lambda x: 'Evening' if x>=18 and x<20 else ('Night' if x>= 20 and x<=23 else('Afternoon' if x>=12 and x<18 else ('Morning' if x>=8 and x<12 else 'Dawn'))))
        
        return df

    @staticmethod
    def holidays(df: pd.DataFrame) -> pd.DataFrame:  
        cal = calendar()
        holidays = cal.holidays(start=df['Dates'].min(), end=df['Dates'].max())
        df['Holiday'] = df['Dates'].dt.date.astype('datetime64').isin(holidays)

        return df

    @staticmethod
    def find_streets(address: str) -> str:
        street_types = ['AV', 'ST', 'CT', 'PZ', 'LN', 'DR', 'PL', 'HY', 
                        'FY', 'WY', 'TR', 'RD', 'BL', 'WAY', 'CR', 'AL', 'I-80',  
                        'RW', 'WK','EL CAMINO DEL MAR']
        street_pattern = '|'.join(street_types)
        streets = re.findall(street_pattern, address)
        if len(streets) == 0:
            return 'OTHER'
        elif len(streets) == 1:
            return streets[0]
        else:
            return 'INT'

    @staticmethod
    def normalization_data(df: pd.DataFrame) -> pd.DataFrame: 
        scaler = MaxAbsScaler()
        data = scaler.fit_transform(df)
        
        return data

    @staticmethod
    def one_hot_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame: 
        ohe = OneHotEncoder()
        for i in columns:
            X = ohe.fit_transform(df[i].values.reshape(-1,1)).toarray()
            dfOneHot = pd.DataFrame(X, columns=[i for i in df[i].unique()])
            df = pd.concat([df, dfOneHot], axis=1)   
            df = df.drop([i], axis=1)
        
        return df

    @staticmethod
    def label_encoder(y: pd.DataFrame) -> pd.DataFrame:
        le = LabelEncoder()
        le.fit(y['Category'].unique())
        y['Category'] = le.transform(y['Category'])
        y = pd.DataFrame(y, columns=['Category'])
        #classes = dict(zip(le.classes_, le.transform(le.classes_)))
        
        return y 

    @staticmethod
    def stratified_sampling(df: pd.DataFrame, percentage: float) -> pd.DataFrame:
        split = StratifiedShuffleSplit(test_size=percentage, random_state=1)
        for _, y in split.split(df, df['Category']):
            df_y = df.iloc[y]

        return df_y

    @staticmethod
    def ordinal_encoding_alphabetically(df: pd.DataFrame, columns: list) -> pd.DataFrame:

        for i in columns:
            categories = {}
            for n, j in enumerate(sorted(list(df[i].unique()))):
                categories[j] = n
            df[i] = df[i].map(categories)
        
        return df