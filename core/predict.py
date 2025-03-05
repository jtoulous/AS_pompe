import os
import json
import yaml
import pickle
import shutil
import pandas as pd
import argparse as ap
import numpy as np
from colorama import Fore, Style

from utils.tools import LoadMetrics, SaveResults
from utils.models import LoadModel 
from utils.filter import FilterDataframe


def Parsing():
    """
    Recupere les arguments d'entree du script.

    Args:
        input du script.

    Return:
        args(obj Argparse): 
            - args.datafile(str): le fichier .csv deja preprocessed sur lequel faire l'inference.
            - args.load(str): le nom du dossier ou les modeles sont sauvegardes.
    """

    parser = ap.ArgumentParser()
    parser.add_argument('config', type=str, help='yaml config file')
    parser.add_argument('-live', action='store_true', default=False, help='real time mode(no batch predictions)')
    return parser.parse_args()



def Predict(df, agent, agent_config, save_repo):
    """
    Fait l'inference.

    Args:
        df (dataframe): Le df sur lequel faire les inferences.
        args (obj Argparse): Les arguments d'entree du script.
        agent (str): Le nom de l'agent.
        agents_variables (dict): Les variables de l'agent(key) et ses predictors(value) tirees du fichier de config.

    Return:
        status (str): green(pas d'anomalie) ou red(anomalie)  
    """

    inference_results = {}
    total_relative_deviation = {}
    total_relative_deviation_sum = 0
    agents_variables = agent_config['variables']

    for variable, features in agents_variables.items():
        model = LoadModel(save_repo, agent, variable) 
        training_rmse, range_dict = LoadMetrics(save_repo, agent, variable)

        X = df[features]
        y_pred = model.predict(X)
        max_val = range_dict['max_value']
        min_val = range_dict['min_value']

        relative_deviation = ((df[variable] - y_pred) / (max_val - min_val)) ** 2

        inference_results[variable] = pd.DataFrame({
            'Date': df['Date'],
            f'Actual_{variable}': df[variable],
            f'Predicted_{variable}': y_pred,
            f'Relative_Deviation_{variable}': relative_deviation
        })

        total_relative_deviation[variable] = relative_deviation.abs().sum()
        total_relative_deviation_sum += total_relative_deviation[variable]

    n_variables = len(agents_variables)
    model_discrepancy = np.sqrt(total_relative_deviation_sum / n_variables)
    status = "green" if model_discrepancy <= 3 else "red"

    weights = {}
    for variable in agents_variables.keys():
        variable_relative_deviation = total_relative_deviation[variable]
        weight = variable_relative_deviation / total_relative_deviation_sum
        weights[variable] = weight * 100
    weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))

    SaveResults(save_repo, weights, model_discrepancy, inference_results, status, agent)
    
    if status == 'green':
        print(Fore.GREEN + "Status: Green" + Style.RESET_ALL)
    else:
        print(Fore.RED + "Status: Red" + Style.RESET_ALL)
    print(f"Global Model Deviation: {model_discrepancy:.2f}")
    return status



def CheckFirstAnomaly(df, agent, agent_config, save_repo, batch_size=20):
    """
    Fait l'inference par batch si la taille du df le permet, pour localiser precisement le debut de la derive.

    Args:
        df (dataframe): le df.
        args (obj Argparse): Les arguments d'entree du script.
        agent (str): Le nom de l'agent.
        agents_variables (dict): Les variables de l'agent(key) et ses predictors(value) tirees du fichier de config.

    Return:
        None
    """

    if len(df) < 10: # Si y a pas assez de data
        return

    agents_variables = agent_config['variables']

    for _, batch in df.groupby(np.arange(len(df)) // batch_size):
        total_relative_deviation = {}
        total_relative_deviation_sum = 0
        
        for variable, features in agents_variables.items():
            model = LoadModel(save_repo, agent, variable) 
            _, range_dict = LoadMetrics(save_repo, agent, variable)

            X = batch[features]
            y_pred = model.predict(X)
            max_val = range_dict['max_value']
            min_val = range_dict['min_value']

            relative_deviation = ((batch[variable] - y_pred) / (max_val - min_val)) ** 2
            total_relative_deviation[variable] = relative_deviation.abs().sum()
            total_relative_deviation_sum += total_relative_deviation[variable]

        n_variables = len(agents_variables)
        model_discrepancy = np.sqrt(total_relative_deviation_sum / n_variables)
        status = "green" if model_discrepancy <= 3 else "red"

        if status == 'red':
            print(f'\nAnomalie started between {batch['Date'].iloc[0]} - {batch['Date'].iloc[-1]}:')
            for var, deviation in total_relative_deviation.items():
                print(f'  - Deviation {var}  ===>  {deviation}')
            return  # On arrête dès qu'on trouve une anomalie

    print('No anomalies found on batch inferences.')
    return



def InferenceOnHistorical(workflow_config, save_repo):
    status = {}
    print(Fore.BLUE + '\n========  Inference Master  ========' + Style.RESET_ALL)
        
    df_master = pd.read_csv(workflow_config['Master']['df_predict'], sep=';')
    if 'filters' in workflow_config['Master']:
        df_master = FilterDataframe(df_master, 'Master', workflow_config['Master']['filters'], save_repo)

    status['Master'] = Predict(df_master, 'Master', workflow_config['Master'], save_repo)
    if status['Master'] == 'red':
        CheckFirstAnomaly(df_master, 'Master', workflow_config['Master'], save_repo) # Fait les inferences par batch pour trouver le commencement de la derive
            
        # If status is red, make inferences with subagents
        subagents = [agent for agent in workflow_config.keys() if agent != 'Master']
        for agent in subagents:
            print(Fore.BLUE + f'\n========  Inference {agent}  ========' + Style.RESET_ALL)

            df_subagent = pd.read_csv(workflow_config[agent]['df_predict'], sep=';')
            if 'filters' in workflow_config[agent]:
                df_subagent = FilterDataframe(df_subagent, subagents, workflow_config[agent]['filters'], save_repo)

            status[agent] = Predict(df_subagent, agent, workflow_config[agent], save_repo)
            if status[agent] == 'red':
                CheckFirstAnomaly(df_subagent, agent, workflow_config[agent], save_repo)
    return status



def InferenceOnRealTime(workflow_config, save_repo):
    status = {}
    print(Fore.BLUE + '\n========  Inference Master  ========' + Style.RESET_ALL)
        
    df_master = pd.read_csv(workflow_config['Master']['df_predict'], sep=';')
    if 'filters' in workflow_config['Master']:
        df_master = FilterDataframe(df_master, 'Master', workflow_config['Master']['filters'], save_repo)

    status['Master'] = Predict(df_master, 'Master', workflow_config['Master'], save_repo)
    if status == 'red':
        # If status is red, make inferences with subagents
        subagents = [agent for agent in config.keys() if agent != 'Master']
        for agent in subagents:
            print(Fore.BLUE + f'\n========  Inference {agent}  ========' + Style.RESET_ALL)

            df_subagent = pd.read_csv(config[agent]['df_predict'], sep=';')
            if 'filters' in config[agent]:
                df_subagent = FilterDataframe(df_subagent, subagents, workflow_config[agent]['filters'], save_repo)

            status[agent] = Predict(df_subagent, agent, config[agent], save_repo)
    return status



if __name__ == '__main__':
    try:
        args = Parsing()        
        config = None

        # Load conf.yaml
        with open(f'{args.config}', 'r') as config_file:
            config = yaml.safe_load(config_file)
       
        system_config = config['System']
        workflow_config = config['Workflow']
        if args.live:
            status = InferenceOnRealTime(workflow_config, system_config['save_repo'])
        else:
            status = InferenceOnHistorical(workflow_config, system_config['save_repo'])

        with open(f'{system_config['save_repo']}/inference_results/inferences.json', 'w') as results_file:
            json.dump(status, results_file, indent=4)

    except Exception as error:
        print(error)



## CHECKER LA TAILLE DU DF AVANT DE FAIRE LES INFERENCE PAR BATCH
# + ADAPTER LA TAILLE DU BATCH EN FONCTION DE LA TAILLE DU DF