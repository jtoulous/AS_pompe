import pandas as pd
import argparse as ap
from pandas import to_datetime

from sklearn.model_selection import train_test_split

from utils.tools import CreateSaveRepo, ApplyFilters
from utils.models import AvailableModels, CheckModels, GetModels, SaveModel
from utils.metrics import GetMetrics, GetWeights
#from utils.feature_engineering import MotorFeatures, HydraulicsFeatures, ElectricsFeatures





def Parsing():
    parser = ap.ArgumentParser()
    parser.add_argument('datafile', type=str, help='csv datafile')
    parser.add_argument('save', type=str, help='name of the folder to save models and results')
    parser.add_argument('-models', nargs='+', type=str, default=AvailableModels(), help='models to be tested')
    parser.add_argument('-filters', action='store_true', default=False, help='apply filters')
    parser.add_argument('-grid_search', action='store_true', default=False, help='activate grid search')
    args = parser.parse_args()
    
#    if args.filters is True:
#        args.filters = GetFilters()

    if CheckModels(args.models) is False:
        raise Exception('Error: one of the models chosen is not available int this prog')
    return args




def TrainModels(df, args):
    numeric_columns = df.select_dtypes(include=['number']).columns
    results = {}
    best_model = {'model': None, 'results': {'train': {}, 'test': {}}}

    for label in numeric_columns:
        features = [col for col in numeric_columns if col != label]
        X_train, X_test, y_train, y_test = train_test_split(df[features], df[label], test_size=0.2, random_state=42)
        
        if args.grid_search is False:
            models_to_test = GetModels(args.models)
        else:
            models_to_test = GetModels(args.models, grid_search=True, X=X_train, y=y_train)

        for model in models_to_test:
            model_name = model.__class__.__name__
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
            test_results.to_csv(f'data/{args.save}/results/{label}/test_results.csv', sep=';', index=False)

            if metrics_test['R2'] > best_model['results']['test'].get('R2', -float('inf')):
                best_model['model'] = model
                best_model['results'] = results[model_name]
        SaveModel(args.save, label, best_model, results)




#def TrainSubModels(sub_type, df, args):




if __name__ == '__main__':
    try:
        args = Parsing()

        df = ApplyFilters(pd.read_csv(args.datafile, sep=';'), args.filters)
#        df_motor = MotorFeatures(df)
#        df_hydraulics = HydraulicsFeatures(df)
#        df_electrics = ElectricsFeatures(df)

        CreateSaveRepo(df, args.save)

        TrainModels(df, args)
#        TrainSubModels('motor', df_motor, args)
#        TrainSubModels('hydraulics', df_hydraulics, args)
#        TrainSubModels('electrics', df_electrics, args)



    except Exception as error:
        print(error)