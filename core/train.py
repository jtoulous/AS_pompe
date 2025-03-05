import os
import logging
import yaml
import warnings
import pandas as pd
import argparse as ap
import shutil
from pandas import to_datetime
from colorama import Fore, Style

from sklearn.metrics import r2_score

from utils.tools import CreateSaveRepo, SplitTrainDf
from utils.models import SaveModel, InitModel, GetModelsToTest
from utils.metrics import GetMetrics, GetWeights
from utils.filter import FilterDataframe
from utils.preprocess import PreprocessByType


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



def Parsing():
    """
    Recupere les arguments d'entree du script.

    Args:
        input du script.

    Return:
        args(obj Argparse): 
            - args.config(str): le fichier .yaml avec la structure des agents.
            - args.grid_search(Booleen): pour faire un grid_search des meilleurs hyper-parametres de chaque modele a tester. 
    """
    
    parser = ap.ArgumentParser()
    parser.add_argument('config', type=str, help='yaml config file') 
#    parser.add_argument('-grid_search', action='store_true', default=False, help='activate grid search')
    
    return parser.parse_args()



def TrainSemiSupervised(df, agent, agents_config, save_repo):
    """
    Entrainement semi supervise de l'agent.

    Args:
        df (dataframe): Le df d'entrainement.
        agent (str): Le nom de l'agent.
        agent_config (dict): Config de l'agent

    Return:
        None  
    """

    df = df.copy()
    results = {}
    model_type = agent_config['model_type']
    agents_variables = agents_config['variables']
#    save_repo = agent_config['save_repo']

    # Iteration sur chaque colonne
    for variable, predictors in agents_variables.items():
        best_model = {'model': None, 'results': {'train': {}, 'test': {}}}

        # Split X and y, according to model_type
        df = PreprocessByType(df, model_type)
        X_train, X_test, y_train, y_test = SplitTrainDf(df, variable, predictors, model_type)

        logging.info(Fore.GREEN + f'===================   Training {variable}  ===================' + Style.RESET_ALL)
        logging.info(f'predictors: {predictors}')

        # Iteration sur chaque modele a tester
        for model_name in GetModelsToTest(model_type):
            logging.info(Fore.LIGHTBLUE_EX + f'  Testing {model_name}...' + Style.RESET_ALL)
            
            model = InitModel(model_name)
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Recuperation des metrics d'entrainement et de test
            metrics_train = GetMetrics(y_train, y_pred_train)
            metrics_test = GetMetrics(y_test, y_pred_test)
            predictors_weights = GetWeights(model, predictors)

            min_value = df[variable].min()
            max_value = df[variable].max()

            results[model_name] = {
                'train': {
                    **metrics_train,
                    'predictors_weights': predictors_weights,
                },
                'test': {
                    **metrics_test,
                    'predictors_weights': predictors_weights,
                },
                'range': {
                    'min_value': float(min_value),
                    'max_value': float(max_value)
                }
            }

            # Sauvegarde des inferences sur le test dans un df
            test_results = pd.DataFrame({
                'Date': df.loc[X_test.index, 'Date'].apply(to_datetime),
                'True': y_test,
                'Predicted': y_pred_test
            })
            test_results.to_csv(f'{save_repo}/{agent}/results/{variable}/test_results.csv', sep=';', index=False)

            logging.info(Fore.LIGHTBLUE_EX + f'   ==> r2: {metrics_test['R2']}' + Style.RESET_ALL)

            # On garde les metrics du meilleur modele
            if metrics_test['R2'] > best_model['results']['test'].get('R2', -float('inf')):
                best_model['model'] = model
                best_model['results'] = results[model_name]

        logging.info(f'{best_model['model'].__class__.__name__} ==> {best_model['results']['test']['R2']}')

        # Sauvegarde du meilleur modele et des metrics
        SaveModel(save_repo, variable, best_model, results, agent)




if __name__ == '__main__':
    try:
        args = Parsing()
        config = None
       
        # Load yaml config file
        with open(args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
 
        system_config = config['System']
        workflow_config = config['Workflow']
        for agent_name, agent_config in workflow_config.items():
            # Create save repo
            CreateSaveRepo(system_config['save_repo'], agent_name, agent_config['variables'])
            
            # Load dataframe + apply filters si necessaire
            df = pd.read_csv(agent_config['df_train'], sep=';')
            if 'filters' in agent_config:
                df = FilterDataframe(df, agent_name, agent_config['filters'], system_config['save_repo'])
            
            # Train according to training_type
            if agent_config['training_type'] == 'Semi Supervised':
                TrainSemiSupervised(df, agent_name, agent_config, system_config['save_repo'])
            
#            elif agent_config['training_type'] == 'Supervised':
#                TrainSupervised(df, agent_name, agent_config)

#            elif agent_config['training_type'] == 'Unsupervised':
#                TrainUnsupervised(df, agent_name, agent_config)


    except Exception as error:
        print(error)