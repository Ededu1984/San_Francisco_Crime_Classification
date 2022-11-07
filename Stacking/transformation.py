import pandas as pd
from utils import Transformation
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # Reading dataset
    train = pd.read_csv("../datasets/train.csv")
    print("-"*80)

    # Dtype transformation
    train['Dates'] = pd.to_datetime(train['Dates'])

    # Droppping the columns Descript and Resolution
    train = train.drop(['Descript','Resolution'], axis=1)
    print("The columns Descript and Resolution were dropped")

    # Preprocessing the data
    train = Transformation.preprocess_date(train)
    print("The data has been processed")

    # Including the season
    train['Season'] = train.Dates.map(Transformation.season_of_date)
    print("The season has been included")

    # Including the period of the day
    train = Transformation.day_period(train)
    print("The period of the day has been included")

    # Including the holiday column
    train = Transformation.holidays(train)
    print("The holiday column was created")

    # Additional transformations
    train['StreetType'] = train['Address'].map(Transformation.find_streets)
    train['Holiday'] = train['Holiday'].apply(lambda x: 1 if x == True else 0)
    X = train.drop(['Category', 'Dates', 'Address'], axis=1)
    y = train[['Category']]
    print("Additional transformations finished")

    # Encoding
    y= Transformation.label_encoder(y)
    X = Transformation.one_hot_encoding(X, ['PdDistrict','StreetType'])
    X = Transformation.ordinal_encoding_alphabetically(X, ['DayOfWeek', 'Year', 'Month', 'Day', 'Season', 'Day_period'])
    X['X'] = X['X'].apply(lambda x: abs(x))
    print("The encoding has been finished")

    # Concatenating the data
    train = pd.concat([X, y], axis=1)
    # Saving the dataset
    train.to_csv("../datasets/train_encoded.csv", index=False)
    print("The csv file was saved")
    print("-"*80)