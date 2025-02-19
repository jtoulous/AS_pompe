import os
import pandas as pd
import argparse as ap

import json
import numpy as np
import pickle
import altair as alt

from utils.tools import LoadModel, LoadMetrics
from utils.feature_engineering import MotorFeatures, HydraulicsFeatures, ElectricsFeatures


def Parsing():
    parser = ap.ArgumentParser()
    parser.add_argument('datafile', type=str, help='csv datafile')
    parser.add_argument('load', type=str, help='repo with saved models')

    args = parser.parse_args()
    args.load = os.path.join(os.getcwd(), 'data', args.load)
    return args


def Predict(df, args, agent):
    numeric_columns = df.select_dtypes(include='number').columns
    inference_results = {}
    total_relative_deviation = {}
    total_relative_deviation_sum = 0

    for label in numeric_columns:
        model = LoadModel(args.load, agent, label) 
        training_rmse, range_dict = LoadMetrics(args.load, agent, label)

        features = [col for col in numeric_columns if col != label]
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

        total_relative_deviation[label] = relative_deviation.abs().sum()
        total_relative_deviation_sum += total_relative_deviation[label]

    n_variables = len(numeric_columns)
    model_discrepancy = np.sqrt(total_relative_deviation_sum / n_variables)
    status = "green" if model_discrepancy <= 3 else "red"

    weights = {}
    for variable in numeric_columns:
        variable_relative_deviation = total_relative_deviation[variable]
        weight = variable_relative_deviation / total_relative_deviation_sum
        weights[variable] = weight * 100
    weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))

    SaveResults(args.load, weights, model_discrepancy, status, agent)
    return status



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

    print(f"Global Model Deviation: {model_discrepancy:.2f}")
    print(f"Status: {'green' if status == 'green' else 'red'}")



if __name__ == '__main__':
    try:
        args = Parsing()        
        
        df = pd.read_csv(args.datafile, sep=';')
#        df_master = LoadFilters(df, args.load)
        df_master = df

        print('\n========  Inference Master  ========')
        status = Predict(df_master, args, 'Master')

        # Si Master detecte de l'anomalie, on regarde quelle zone presente le plus d'anomalie
        if status == 'red':
            df_motor = MotorFeatures(df_master)
            df_hydraulics = HydraulicsFeatures(df_master)
            df_electrics = ElectricsFeatures(df_master)

            print('\n========  Inference Motor  ========')
            Predict(df_motor, args, 'Motor')

            print('\n========  Inference Hydraulics  ========')
            Predict(df_hydraulics, args, 'Hydraulics')

            print('\n========  Inference Electrics  ========')
            Predict(df_electrics, args, 'Electrics')


    except Exception as error:
        print(error)