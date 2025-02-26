import os
import json
import yaml
import pickle
import shutil
import pandas as pd
import argparse as ap
import numpy as np
import altair as alt
from colorama import Fore, Style

from utils.tools import LoadModel, LoadMetrics
from utils.filter import LoadFilters


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
    parser.add_argument('datafile', type=str, help='csv datafile to predict')
    parser.add_argument('load', type=str, help='repo with saved models')

    args = parser.parse_args()
    args.load = os.path.join(os.getcwd(), 'data', args.load)

    # Suppression des anciennes inferences
    if os.path.exists(f'{args.load}/inference_results'):
        shutil.rmtree(f'{args.load}/inference_results')
    
    return args



def SaveResults(save_repo, weights, model_discrepancy, inference_results, status, agent):
    """
    Sauvegarde les resultat de l'inference.

    Args:
        save_repo (str): le path du dossier de l'equipement.
        weights (dict): les poids des variables, {variable: poid}.
        model_discrepancy (float): le discrepancy du model.
        inference_results (dict): les resultat des inferences et les vrais valeurs pour chaque variable dans un df, {variable: df}
        status(str): green ou red.
        agent(str): nom de l'agent.

    Return:
        None    
    """
    output_folder = f'{save_repo}/inference_results/{agent}'
    os.makedirs(output_folder, exist_ok=True)

    # Save les dfs d'inferences
    for label, df in inference_results.items():
        df.to_csv(f'{output_folder}/{label}_results.csv', sep=';', index=None)

    results = {
        "weights": weights,
        "model_discrepancy": model_discrepancy,
        "status": status
    }

    # Save les resultats
    with open(f'{output_folder}/results.json', "w") as f:
        json.dump(results, f, indent=4)

    # Creation des charts avec les weights
    CreatePlot(output_folder, weights, model_discrepancy, status)



def CreatePlot(save_repo, weights, model_discrepancy, status):
    """
    Cree le plot des poids.

    Args:
        save_repo (str): le path du dossier de l'equipement.
        weights (dict): les poids des variables, {variable: poid}.
        model_discrepancy (float): le discrepancy du modele.
        status(str): green ou red.

    Return:
        None
    """

    # Créer un DataFrame à partir des poids
    weights_df = pd.DataFrame(list(weights.items()), columns=["Variable", "Weight"])

    # Créer un graphique interactif avec Altair
    chart = alt.Chart(weights_df).mark_bar().encode(
        x=alt.X('Variable:N', sort='-y', title='Variable'),
        y=alt.Y('Weight:Q', title='Weight (%)'),
        tooltip=['Variable', 'Weight']
    ).properties(
        title='Variable Weights',
        width=800,
        height=400
    ).interactive()
    
    chart.save(f'{save_repo}/weights_chart.html')

    if status == 'green':
        print(Fore.GREEN + "Status: Green" + Style.RESET_ALL)
    else:
        print(Fore.RED + "Status: Red" + Style.RESET_ALL)
    print(f"Global Model Deviation: {model_discrepancy:.2f}")



def Predict(df, args, agent, agents_variables):
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

    for label, features in agents_variables.items():
        model = LoadModel(args.load, agent, label) 
        training_rmse, range_dict = LoadMetrics(args.load, agent, label)

        X = df[features]
        y_pred = model.predict(X)
        max_val = range_dict['max_value']
        min_val = range_dict['min_value']

        relative_deviation = ((df[label] - y_pred) / (max_val - min_val)) ** 2

        inference_results[label] = pd.DataFrame({
            'Date': df['Date'],
            f'Actual_{label}': df[label],
            f'Predicted_{label}': y_pred,
            f'Relative_Deviation_{label}': relative_deviation
        })

        # a rajouter: save le df

        total_relative_deviation[label] = relative_deviation.abs().sum()
        total_relative_deviation_sum += total_relative_deviation[label]

    n_variables = len(agents_variables)
    model_discrepancy = np.sqrt(total_relative_deviation_sum / n_variables)
    status = "green" if model_discrepancy <= 3 else "red"

    weights = {}
    for label in agents_variables.keys():
        label_relative_deviation = total_relative_deviation[label]
        weight = label_relative_deviation / total_relative_deviation_sum
        weights[label] = weight * 100
    weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))

    SaveResults(args.load, weights, model_discrepancy, inference_results, status, agent)
    return status



def CheckFirstAnomaly(df, args, agent, agents_variables, batch_size=20):
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
    for _, batch in df.groupby(np.arange(len(df)) // batch_size):
        total_relative_deviation = {}
        total_relative_deviation_sum = 0
        
        for label, features in agents_variables.items():
            model = LoadModel(args.load, agent, label) 
            _, range_dict = LoadMetrics(args.load, agent, label)

            X = batch[features]
            y_pred = model.predict(X)
            max_val = range_dict['max_value']
            min_val = range_dict['min_value']

            relative_deviation = ((batch[label] - y_pred) / (max_val - min_val)) ** 2
            total_relative_deviation[label] = relative_deviation.abs().sum()
            total_relative_deviation_sum += total_relative_deviation[label]

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





if __name__ == '__main__':
    try:
        args = Parsing()        
        config = None

        # Load dataframe + filter it
        df = pd.read_csv(args.datafile, sep=';')
        df = LoadFilters(df, args.load)

        # Load conf.yaml
        with open(f'{args.load}/conf.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)

        # Master agent predicts
        print(Fore.BLUE + '\n========  Inference Master  ========' + Style.RESET_ALL)
        status = Predict(df, args, 'Master', config['Master'])
        
        if status == 'red':
            CheckFirstAnomaly(df, args, 'Master', config['Master']) # Fait les inferences par batch pour trouver le commencement de la derive

            # If status is red, make inferences with subagents
            subagents = [agent for agent in config.keys() if agent != 'Master']
            for agent in subagents:
                print(Fore.BLUE + f'\n========  Inference {agent}  ========' + Style.RESET_ALL)
                status = Predict(df, args, agent, config[agent])
                if status == 'red':
                    CheckFirstAnomaly(df, args, agent, config[agent])


    except Exception as error:
        print(error)



## CHECKER LA TAILLE DU DF AVANT DE FAIRE LES INFERENCE PAR BATCH
# + ADAPTER LA TAILLE DU BATCH EN FONCTION DE LA TAILLE DU DF