import os
import json
import shutil
import logging
import pickle
import pandas as pd


def CreateSaveRepo(save_repo, agent, variables):
    logging.info('Creating save repository...')
    
    # Suppression de l'ancien repo si il existe deja
    if os.path.exists(save_repo) and agent == 'Master':
        shutil.rmtree(save_repo)
    
    os.makedirs(save_repo, exist_ok=True)
    os.makedirs(f'{save_repo}/models/{agent}', exist_ok=True)
    os.makedirs(f'{save_repo}/results/{agent}', exist_ok=True)

    for var in variables.keys():
        os.makedirs(f'{save_repo}/results/{agent}/{var}', exist_ok=True)


def LoadModel(save_repo, agent, label):
    model_filename = f'{save_repo}/models/{agent}/{label}_best_model.pkl'

    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


def LoadMetrics(save_repo, agent, label):
    metrics_filename = f'{save_repo}/results/{agent}/{label}/metrics.json'

    with open(metrics_filename, 'r') as metrics_file:
        metrics_data = json.load(metrics_file)

    training_rmse = metrics_data.get("best_model", {}).get("test", {}).get("RMSE")
    range_dict = metrics_data.get("best_model", {}).get("range", {})
    return training_rmse, range_dict