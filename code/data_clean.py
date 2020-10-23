import pandas as pd 
import numpy as np

def month_dummies(column):
    """
    takes in a column, slices the entries to the first 8 characters, 
    converts to datetime and returns dummy columns for every month, dropping december
    """ 
    i = column.apply(lambda x: x[0:8])
    i = pd.to_datetime(i)
    i = i.dt.strftime('%m')
    return dummy_list(i, 'mon', 12) 

def new_construction(renovated, built):
    '''takes in a column of renovation years and initial construction and
    returns a new column which includes the date of the renovation if any, 
    else the construction date'''
    new_const = []
    for i in list(range(len(renovated))):
        if renovated[i] != 0:
            new_const.append(renovated[i])
        else:
            new_const.append(built[i]) 
    return new_const

def dummy_list(column, prefix, drop_value):
    ''' Takes in a column, a prefix and drop value and generates a DataFrame of dummy columns with the specified prefix and column categories minus the specified drop value'''
    dum = pd.get_dummies(column, prefix = prefix)
    dum = dum.drop(columns = f'{prefix}_{drop_value}')
    return dum

def best_schools(column, zips):
    schools = []
    for i in column:
        if i in zips:
            schools.append(1)
        else:
            schools.append(0)
    return schools

def price_adjust(column):
    adj = []
    t1 = np.percentile(column, 2.5)
    t2 = np.percentile(column, 97.5)
    for i in column:
        if i < t1:
            adj.append(t1)
        elif i > t2:
            adj.append(t2)
        else:
            adj.append(i)
    return adj