import os
import shutil
import logging
import pandas as pd



def CreateSaveRepo(df, save_repo, model):
    logging.info('Creating save repository...')
    
    if os.path.exists(save_repo) and model == 'main_model':
        shutil.rmtree(save_repo)
    
    os.makedirs(save_repo, exist_ok=True)
    os.makedirs(f'{save_repo}/models/{model}', exist_ok=True)
    os.makedirs(f'{save_repo}/results/{model}', exist_ok=True)

    numeric_vars = df.select_dtypes(include=['number']).columns
    for var in numeric_vars:
        os.makedirs(f'{save_repo}/results/{model}/{var}', exist_ok=True)