import os
import logging
import warnings
import pandas as pd
import argparse as ap
from pandas import to_datetime
from colorama import Fore, Style

from sklearn.model_selection import train_test_split

from utils.tools import CreateSaveRepo
from utils.models import AvailableModels, CheckModels, GetModels, SaveModel, SaveSubmodel
from utils.metrics import GetMetrics, GetWeights
from utils.filter import ApplyFilters
from utils.feature_engineering import MotorFeatures, HydraulicsFeatures, ElectricsFeatures


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



def Parsing():
    parser = ap.ArgumentParser()
    parser.add_argument('datafile', type=str, help='csv datafile')
    parser.add_argument('save', type=str, help='name of the folder to save models and results')
    parser.add_argument('-models', nargs='+', type=str, default=AvailableModels(), help='models to be tested')
    parser.add_argument('-filters', action='store_true', default=False, help='apply filters')
    parser.add_argument('-grid_search', action='store_true', default=False, help='activate grid search')
    args = parser.parse_args()
    
    args.save = os.path.join(os.getcwd(), 'data', args.save)

    if CheckModels(args.models) is False:
        raise Exception('Error: one of the models chosen is not available int this prog')
    return args




def TrainModels(df, args, submodel=None):
    numeric_columns = df.select_dtypes(include=['number']).columns
    results = {}

    for label in numeric_columns:
        logging.info(Fore.GREEN + f'===================   Training {label}  ===================' + Style.RESET_ALL)
        best_model = {'model': None, 'results': {'train': {}, 'test': {}}}
        features = [col for col in numeric_columns if col != label]
        X_train, X_test, y_train, y_test = train_test_split(df[features], df[label], test_size=0.2, random_state=42)
        
        if args.grid_search is False:
            models_to_test = GetModels(args.models)
        else:
            models_to_test = GetModels(args.models, grid_search=True, X=X_train, y=y_train)

        for model in models_to_test:
            model_name = model.__class__.__name__
            logging.info(Fore.LIGHTBLUE_EX + f'  Testing {model_name}...' + Style.RESET_ALL)
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            metrics_train = GetMetrics(y_train, y_pred_train)
            metrics_test = GetMetrics(y_test, y_pred_test)

            features_weights = GetWeights(model, features)

            min_value = df[label].min()
            max_value = df[label].max()

            results[model_name] = {
                'train': {
                    **metrics_train,
                    'features_weights': features_weights,
                },
                'test': {
                    **metrics_test,
                    'features_weights': features_weights,
                },
                'range': {
                    'min_value': float(min_value),
                    'max_value': float(max_value)
                }
            }

            test_results = pd.DataFrame({
                'Date': df.loc[X_test.index, 'Date'].apply(to_datetime),
                'True': y_test,
                'Predicted': y_pred_test
            })
            if submodel is None:
                test_results.to_csv(f'{args.save}/results/main_model/{label}/test_results.csv', sep=';', index=False)
            else:
                test_results.to_csv(f'{args.save}/results/{submodel}/{label}/test_results.csv', sep=';', index=False)

            logging.info(Fore.LIGHTBLUE_EX + f'   ==> r2: {metrics_test['R2']}' + Style.RESET_ALL)
            if metrics_test['R2'] > best_model['results']['test'].get('R2', -float('inf')):
                best_model['model'] = model
                best_model['results'] = results[model_name]
        
        logging.info(f'{best_model['model'].__class__.__name__} ==> {best_model['results']['test']['R2']}')
        if submodel is None:
            SaveModel(args.save, label, best_model, results)
        else:
            SaveSubmodel(args.save, label, best_model, results, submodel)



#def TrainSubModels(sub_type, df, args):




if __name__ == '__main__':
    try:
        args = Parsing()

        logging.info('Reading data...')
        df = pd.read_csv(args.datafile, sep=';')

        df = ApplyFilters(df, args.filters, args.save)
        df_motor = MotorFeatures(df)
        df_hydraulics = HydraulicsFeatures(df)
        df_electrics = ElectricsFeatures(df)

        CreateSaveRepo(df, args.save, 'main_model')
        CreateSaveRepo(df_motor, args.save, 'Motor')
        CreateSaveRepo(df_hydraulics, args.save, 'Hydraulics')
        CreateSaveRepo(df_electrics, args.save, 'Electrics')

        TrainModels(df, args)
        TrainModels(df_motor, args, submodel='Motor')
        TrainModels(df_hydraulics, args, submodel='Hydraulics')
        TrainModels(df_electrics, args, submodel='Electrics')



    except Exception as error:
        print(error)