import os
import logging
import pickle
import json

# Semi supervised Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# Time series models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
#from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX
#from statsmodels.tsa.holtwinters import ExponentialSmoothing
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM


def RegressionModels():
    available_models = [
        'LinearRegression',
        'Ridge',
        'Lasso',
        'ElasticNet',
        'DecisionTreeRegressor',
        'RandomForestRegressor',
        'GradientBoostingRegressor',
        'AdaBoostRegressor',
        'BaggingRegressor',
        'SVR',
        'KNeighborsRegressor'
    ]
    return available_models


def TimeSeriesModels():
    available_models = [
#        'ARIMA',
#        'SARIMAX',
#        'ExponentialSmoothing',
        'XGBoost',
        'LightGBM',
        'CatBoost',
#        'LSTM',
    ]
    return available_models



def GetModelsToTest(agent_type):
    if agent_type == 'Regression':
        return RegressionModels()
    elif agent_type == 'Time Series':
        return TimeSeriesModels()
    else:
        raise Exception(f'Error: {agent_type} no such model type.')



def InitModel(model_name, variable=None, predictors=None):
    # Regression models
    if model_name == 'LinearRegression':
        return LinearRegression()
    
    elif model_name == 'Ridge':
        return Ridge()
    
    elif model_name == 'Lasso':
        return Lasso()
    
    elif model_name == 'ElasticNet':
        return ElasticNet()
    
    elif model_name == 'DecisionTreeRegressor':
        return DecisionTreeRegressor()
    
    elif model_name == 'RandomForestRegressor':
        return RandomForestRegressor()
    
    elif model_name == 'GradientBoostingRegressor':
        return GradientBoostingRegressor()
    
    elif model_name == 'AdaBoostRegressor':
        return AdaBoostRegressor()
    
    elif model_name == 'BaggingRegressor':
        return BaggingRegressor()
    
    elif model_name == 'SVR':
        return SVR()
    
    elif model_name == 'KNeighborsRegressor':
        return KNeighborsRegressor()


    # Time Series Models
    if model_name == 'ARIMA':
        return ARIMA(variable, order=(1, 1, 1))

    elif model_name == 'SARIMAX':
        return SARIMAX(variable, exog=predictors, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

    elif model_name == 'ExponentialSmoothing':
        return ExponentialSmoothing(variable, trend='add', seasonal=None)

    elif model_name == 'XGBoost':
        return XGBRegressor()

    elif model_name == 'LightGBM':
        return LGBMRegressor(verbose=-1)

    elif model_name == 'CatBoost':
        return CatBoostRegressor(verbose=0)

    elif model_name == 'LSTM':
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(None, 1)),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model



def SaveModel(save_repo, variable, best_model, results, agent):
    logging.info(f'Saving best model and metrics for {variable}...\n')
    with open(f'{save_repo}/{agent}/models/{variable}_best_model.pkl', 'wb') as model_file:
        pickle.dump(best_model['model'], model_file)

    with open(f'{save_repo}/{agent}/results/{variable}/metrics.json', 'w') as metrics_file:
        json.dump({'best_model': best_model['results'], 'all_models': results}, metrics_file)


def LoadModel(save_repo, agent, variable):
    model_filename = f'{save_repo}/{agent}/models/{variable}_best_model.pkl'

    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    return model














