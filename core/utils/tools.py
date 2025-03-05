import os
import json
import shutil
import logging
import pickle
import pandas as pd
import altair as alt

from sklearn.model_selection import train_test_split


def CreateSaveRepo(save_repo, agent, variables):
    logging.info('Creating save repository...')
    
    if os.path.exists(f'{save_repo}/{agent}'):
        shutil.rmtree(f'{save_repo}/{agent}')

    os.makedirs(f'{save_repo}/{agent}/models', exist_ok=True)
    for var in variables.keys():
        os.makedirs(f'{save_repo}/{agent}/results/{var}', exist_ok=True)


def LoadMetrics(save_repo, agent, variable):
    metrics_filename = f'{save_repo}/{agent}/results/{variable}/metrics.json'

    with open(metrics_filename, 'r') as metrics_file:
        metrics_data = json.load(metrics_file)

    training_rmse = metrics_data.get("best_model", {}).get("test", {}).get("RMSE")
    range_dict = metrics_data.get("best_model", {}).get("range", {})
    return training_rmse, range_dict


def SplitTrainDf(df, variable, predictors, agent_type):
    if agent_type == 'Regression':
        X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[variable], test_size=0.2, random_state=42)

    elif agent_type == 'Time Series':
        split_idx = int(0.8 * len(df))
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]
        X_train, X_test, y_train, y_test = df_train[predictors], df_test[predictors], df_train[variable], df_test[variable]
    
    return X_train, X_test, y_train, y_test


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