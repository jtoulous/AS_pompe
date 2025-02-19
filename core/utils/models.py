import os
import logging
import pickle
import json

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV



def GetModels(models_wanted, grid_search=False, X=None, y=None): # RAJOUTER LE GRID SEARCH EVENTUELLEMENT
    logging.info(f'Initializing models...')
    models = []
    for model in models_wanted:
        models.append(InitModel(model))

    if grid_search == True:
        for i in range(len(models)):
            models[i] = GridSearch(models[i], X, y)

    return models



def AvailableModels():
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


def CheckModels(chosen_models):
    available_models = AvailableModels()
    for model in chosen_models:
        if model not in available_models:
            return False
    return True


def InitModel(model_name):
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


def GridSearch(model, X_train, y_train):
    logging.info(f'Grid searching for {model.__class__.__name__}...')        
    if model.__class__.__name__ == 'LinearRegression':
        return model

    param_grid = GetGrid(model.__class__.__name__)

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    model_with_best_params = model.__class__(**best_params)
    logging.info(f'  ==> best params: {best_params}')
    return model_with_best_params



# Get params to test
def GetGrid(model_name):
    param_grid = {}
    if model_name == 'LinearRegression':
        param_grid = {} # Pas d'hyperparamètres à régler pour la régression linéaire standard
    
    elif model_name == 'Ridge':
        param_grid = {
            'alpha': [0.1, 1.0, 10.0]
        }
    
    elif model_name == 'Lasso':
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    
    elif model_name == 'ElasticNet':
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        }
    
    elif model_name == 'DecisionTreeRegressor':
        param_grid = {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    
    elif model_name == 'RandomForestRegressor':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
    
    elif model_name == 'GradientBoostingRegressor':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    
    elif model_name == 'AdaBoostRegressor':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        }
    
    elif model_name == 'BaggingRegressor':
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_samples': [0.5, 0.75, 1.0]
        }
    
    elif model_name == 'SVR':
        param_grid = {
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 0.5]
        }
    
    elif model_name == 'KNeighborsRegressor':
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    
#    elif model_name == 'XGBoost':
#        param_grid = {
#            'n_estimators': [100, 200, 300],
#            'learning_rate': [0.01, 0.1, 0.2],
#            'max_depth': [3, 5, 7]
#        }
    return param_grid




def SaveModel(save_repo, label, best_model, results, agent):
    logging.info(f'Saving best model and metrics for {label}...\n')
    with open(f'{save_repo}/models/{agent}/{label}_best_model.pkl', 'wb') as model_file:
        pickle.dump(best_model['model'], model_file)

    with open(f'{save_repo}/results/{agent}/{label}/metrics.json', 'w') as metrics_file:
        json.dump({'best_model': best_model['results'], 'all_models': results}, metrics_file)
















