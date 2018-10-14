##importing important libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

##Reading the data set to understand what the data looks like.

my_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print(my_data.head()) ## Having a look at the raw data 
print(my_data.columns) ## Looking at the different columns and exploring the dataset
my_data.Sex = my_data.Sex.replace({'male':1, 'female': 0})
my_data.Age = my_data.Age.fillna(my_data.Age.median())
test_data.Sex = test_data.Sex.replace({'male':1, 'female': 0})
test_data.Age = test_data.Age.fillna(test_data.Age.median())

## After getting the insights, Deciding on the factors and cleaning the data

Y_training = my_data.Survived
X_cols = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']
x_train = my_data[X_cols]
x_test = test_data[X_cols]


def get_null_percentage(data_frame, threshold):
    list_of_columns = data_frame.columns
    data_frame[list_of_columns]
    dict_null = {}
    for i in range(len(list_of_columns)):
        x = data_frame[list_of_columns[i]]
        per_null = (x.isnull().sum()/len(data_frame)) * 100
        if per_null >= threshold:
            dict_null[list_of_columns[i]] = per_null
    return(dict_null)


x_test.Fare = x_test.Fare.fillna(x_test.Fare.median())

## Random_forest_regressor
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(x_train, Y_training)
survived_series = forest_model.predict(x_test)
survival = []
for i in range(len(survived_series)):
    survival.append(int(survived_series[i]))
output = pd.DataFrame({'Passengerid': test_data.PassengerId, 'Survived' : survival })
output.to_csv('Submission_Randomforest.csv', index = False)