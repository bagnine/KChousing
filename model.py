from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def train_model(features, target):
    """
    Takes in a DataFrame of features and a Series of target variables, fits to a linear regression model, makes predictions, compares to the target variable and returns R squared and RSME values
    """
    lr = LinearRegression()
    lr.fit(features, target)
    y_pred = lr.predict(features)
    r2 = lr.score(features, target)
    rsme = mean_squared_error(target, y_pred)
    return print('R Squared:' + str(r2), 'RSME:' + str((rsme**.5)))

def test_model(features, target):
    """Assumes a linear regression model is already in place, returns R squared and RSME values for the model run on a test set.
    """
    y_test_pred = lr.predict(features)
    r2 = lr.score(features, target)
    rsme = mean_squared_error(target, y_test_pred)
    return print('R Squared:' + str(r2), 'RSME:' + str((rsme**.5)))

def polynomialize(features, power):
    """takes in a DataFrame of features and an exponent value, outputs a new DataFrame with PolynomialFeatures applied
    """
    poly = PolynomialFeatures(degree = power)
    featpoly = poly.fit_transform(features)
    poly_columns = poly.get_feature_names(features.columns)
    return pd.DataFrame(featpoly, columns=poly_columns)