import os
import json
import yaml
import pickle
import pandas as pd
import argparse as ap
import numpy as np
import altair as alt
from colorama import Fore, Style

from utils.tools import LoadModel, LoadMetrics
from utils.feature_engineering import MotorFeatures, HydraulicsFeatures, ElectricsFeatures


def Parsing():
    parser = ap.ArgumentParser()
    parser.add_argument('datafile', type=str, help='csv datafile to predict')
    parser.add_argument('load', type=str, help='repo with saved models')

    args = parser.parse_args()
    args.load = os.path.join(os.getcwd(), 'data', args.load)
    return args



def SaveResults(save_repo, weights, model_discrepancy, status, agent):
    output_folder = f'{save_repo}/prediction_results'
    os.makedirs(output_folder, exist_ok=True)

    results = {
        "weights": weights,
        "model_discrepancy": model_discrepancy,
        "status": status
    }

    with open(f'{output_folder}/{agent}_results.json', "w") as f:
        json.dump(results, f, indent=4)

    CreatePlot(output_folder, weights, model_discrepancy, status, agent)



def CreatePlot(save_repo, weights, model_discrepancy, status, agent):
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
    
    chart.save(f'{save_repo}/{agent}_chart.html')

    if status == 'green':
        print(Fore.GREEN + "Status: Green" + Style.RESET_ALL)
    else:
        print(Fore.RED + "Status: Red" + Style.RESET_ALL)
    print(f"Global Model Deviation: {model_discrepancy:.2f}")



def Predict(df, args, agent, agent_config):
    inference_results = {}
    total_relative_deviation = {}
    total_relative_deviation_sum = 0

    for label, features in agent_config.items():
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

    n_variables = len(agent_config)
    model_discrepancy = np.sqrt(total_relative_deviation_sum / n_variables)
    status = "green" if model_discrepancy <= 3 else "red"

    weights = {}
    for label in agent_config.keys():
        label_relative_deviation = total_relative_deviation[label]
        weight = label_relative_deviation / total_relative_deviation_sum
        weights[label] = weight * 100
    weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))

    SaveResults(args.load, weights, model_discrepancy, status, agent)
    return status



#def LoadFilters(df, equipment_folder):
#    metadata_path = os.path.join(equipment_folder, "metadata.json")
#    if os.path.exists(metadata_path):
#        with open(metadata_path, 'r') as metadata_file:
#            metadata = json.load(metadata_file)
#            filters_dict = metadata.get("filters", {})
#            if filters_dict:
#                for var, (op, thresh) in filters_dict.items():
#                    if op == "Greater than":
#                        df = df[df[var] > thresh]
#                    elif op == "Less than":
#                        df = df[df[var] < thresh]
#                    else:
#                        df = df[df[var] == thresh]
#    return df




def CheckFirstAnomaly(df, args, agent, agent_config, batch_size=100):
    for _, batch in df.groupby(np.arange(len(df)) // batch_size):
        total_relative_deviation = {}
        total_relative_deviation_sum = 0
        
        for label, features in agent_config.items():
            model = LoadModel(args.load, agent, label) 
            _, range_dict = LoadMetrics(args.load, agent, label)

            X = batch[features]
            y_pred = model.predict(X)
            max_val = range_dict['max_value']
            min_val = range_dict['min_value']

            relative_deviation = ((batch[label] - y_pred) / (max_val - min_val)) ** 2
            total_relative_deviation[label] = relative_deviation.abs().sum()
            total_relative_deviation_sum += total_relative_deviation[label]

        n_variables = len(agent_config)
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
#        df = LoadFilters(df, args.load)

        # Load conf.yaml
        with open(f'{args.load}/conf.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)

        # Master agent predicts
        print(Fore.BLUE + '\n========  Inference Master  ========' + Style.RESET_ALL)
        status = Predict(df, args, 'Master', config['Master'])
        if status == 'red':
            CheckFirstAnomaly(df, args, 'Master', config['Master']) # Fait les inferences par batch pour trouver le commencement de la derive

            subagents = [agent for agent in config.keys() if agent != 'Master']
            for agent in subagents:
                print(Fore.BLUE + f'\n========  Inference {agent}  ========' + Style.RESET_ALL)
                status = Predict(df, args, agent, config[agent])
                if status == 'red':
                    CheckFirstAnomaly(df, args, agent, config[agent]) # Fait les inferences par batch pour trouver le commencement de la derive


    except Exception as error:
        print(error)