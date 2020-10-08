# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 22:41:24 2020

@author: 91956
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import metrics

beer_df_orignal= pd.read_csv('train.csv')

beer_df= beer_df_orignal.copy()
print(beer_df.columns)

beer_df.drop(['review/timeStruct', 'review/timeUnix', 'user/ageInSeconds', 'user/birthdayRaw',\
       'user/birthdayUnix', 'user/gender', 'user/profileName'], axis=1, inplace= True)

# sum(beer_df['beer/ABV'].isnull())
print(beer_df.isnull().sum())

beer_df['beer/name'].value_counts()
len(beer_df['beer/name'].unique().tolist())
len(beer_df['beer/style'].unique().tolist())


sns.displot(beer_df['beer/ABV'], kde= False, bins= 100)
sns.histplot(beer_df['beer/ABV'])


X= beer_df.drop(['index', 'review/overall', 'review/text', 'beer/beerId', 'beer/brewerId', 'beer/name'], axis=1)
y= beer_df['review/overall']

categorical_cols= ['beer/style']
X= pd.get_dummies(X, columns= categorical_cols)

# new_X= beer_df.drop(['index', 'review/overall', 'review/text', 'beer/beerId', 'beer/brewerId', 'beer/name', 'beer/style'], axis=1)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.3)

sc = StandardScaler()
X_train.iloc[:, : 5] = sc.fit_transform(X_train.iloc[:, : 5])
X_test.iloc[:, : 5] = sc.transform(X_test.iloc[:, : 5])

print('*********Linear Model*********')
lm= LinearRegression()
lm.fit(X_train, y_train)
# print(lm.score(X_train, y_train))
#print(lm.coef_)

predictions= lm.predict(X_test)
# round(predictions[0],1)
#plt.scatter(y_test, predictions)
#plt.xlabel('Y Test')
#plt.ylabel('Predicted Y')

print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('MAE: ', metrics.mean_absolute_error(y_test, predictions))
print('R2 Score: ', metrics.r2_score(y_test, predictions))

accuracies = cross_val_score(estimator = lm, X = X_train, y = y_train, cv = 5)
print(accuracies.mean(), '\n\n')


# XGBoost Regression
print('*********XGBoost Model*********')
import xgboost

xgb_model = xgboost.XGBRegressor(gamma=0.2,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000)

xgb_model.fit(X_train, y_train)


predictions= xgb_model.predict(X_test)

print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('MAE: ', metrics.mean_absolute_error(y_test, predictions))
print('R2 Score: ', metrics.r2_score(y_test, predictions))

accuracies = cross_val_score(estimator = xgb_model, X = X_train, y = y_train, cv = 5)
print(accuracies.mean(), '\n\n')



print('*********SVM Model*********')
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

predictions= regressor.predict(X_test)

print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('MAE: ', metrics.mean_absolute_error(y_test, predictions))
print('R2 Score: ', metrics.r2_score(y_test, predictions))

accuracies = cross_val_score(estimator = xgb_model, X = X_train, y = y_train, cv = 5)
print(accuracies.mean(), '\n\n')























