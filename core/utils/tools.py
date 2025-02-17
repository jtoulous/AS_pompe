import os
import shutil
import pandas as pd



def CreateSaveRepo(df, save_repo):
    if os.path.exists(f'data/{save_repo}'):
        shutil.rmtree(f'data/{save_repo}')
    
    os.makedirs(f'data/{save_repo}', exist_ok=True)
    os.makedirs(f'data/{save_repo}/models', exist_ok=True)
    os.makedirs(f'data/{save_repo}/results', exist_ok=True)

    numeric_vars = df.select_dtypes(include=['number']).columns
    for var in numeric_vars:
        os.makedirs(f'data/{save_repo}/results/{var}', exist_ok=True)





def ApplyFilters(df, filters):
#    if filters is not False:
        #prompter les filtre + les appliquer..

    return df