import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import linregress
from scipy.stats import pearsonr
from pysal.lib import weights  
from sklearn.metrics import r2_score
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

# config
data_path = "challenge_evaluation_data.csv"

def model_fit(X_train, y_train):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model

def model_train_eval(model, X_data, y_data):
    # define model evaluation method
    cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=13)
    # evaluate model
    scores = cross_val_score(model, X_data, y_data, scoring='r2', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = abs(scores)
    print('r-squared: %.3f (%.3f)' % (scores.mean(), scores.std()) )

def model_predict(model, X_test):
	y_pred = model.predict(X_test)
	return y_pred


training_data = pd.read_csv(data_path)
X_train, X_test, y_train, y_test = train_test_split(training_data.drop(['Land_Value', 'City_Name', 'Neighborhood_FID'], axis=1), training_data['Land_Value'], test_size=0.05)

model = model_fit(X_train, y_train)
y_pred = model_predict(model, X_test)

