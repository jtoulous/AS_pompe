import pickle
import json

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


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


def GetModels(models_wanted, grid_search=False, X=None, y=None): # RAJOUTER LE GRID SEARCH EVENTUELLEMENT
    models = []
    for model in models_wanted:
        if model == 'LinearRegression':
            models.append(LinearRegression())
        
        elif model == 'Ridge':
            models.append(Ridge())
        
        elif model == 'Lasso':
            models.append(Lasso())
        
        elif model == 'ElasticNet':
            models.append(ElasticNet())
        
        elif model == 'DecisionTreeRegressor':
            models.append(DecisionTreeRegressor())
        
        elif model == 'RandomForestRegressor':
            models.append(RandomForestRegressor())
        
        elif model == 'GradientBoostingRegressor':
            models.append(GradientBoostingRegressor())
        
        elif model == 'AdaBoostRegressor':
            models.append(AdaBoostRegressor())
        
        elif model == 'BaggingRegressor':
            models.append(BaggingRegressor())
        
        elif model == 'SVR':
            models.append(SVR())
        
        elif model == 'KNeighborsRegressor':
            models.append(KNeighborsRegressor())
    
    return models
    

def SaveModel(save_repo, label, best_model, results):
    with open(f'data/{save_repo}/models/{label}_best_model.pkl', 'wb') as model_file:
        pickle.dump(best_model['model'], model_file)

    with open(f'data/{save_repo}/results/{label}/metrics.json', 'w') as metrics_file:
        json.dump({'best_model': best_model['results'], 'all_models': results}, metrics_file)