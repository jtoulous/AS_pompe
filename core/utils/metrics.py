import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



def GetMetrics(y, y_pred):
    metrics = {}
    metrics['R2'] = float(r2_score(y, y_pred))
    metrics['RMSE'] = float(np.sqrt(mean_squared_error(y, y_pred)))
    metrics['MAE'] = float(mean_absolute_error(y, y_pred))
    metrics['MAPE'] = float(np.mean(np.abs((y - y_pred) / y)) * 100)
    metrics['MSE'] = float(mean_squared_error(y, y_pred))
    metrics['SMAPE'] = float(2 * np.mean(np.abs(y_pred - y) / (np.abs(y_pred) + np.abs(y))) * 100)
    metrics['NRMSE'] = float(metrics['RMSE'] / (y.max() - y.min()))
    return metrics


def GetWeights(model, features):
    model_name = model.__class__.__name__
    features_weights = None
    if model_name in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
        features_weights = dict(zip(features, model.coef_))
    elif model_name in ['DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor']:
        features_weights = dict(zip(features, model.feature_importances_))
    elif model_name == 'BaggingRegressor':
        base_estimator_weights = np.mean([estimator.feature_importances_ for estimator in model.estimators_], axis=0)
        features_weights = dict(zip(features, base_estimator_weights))
    elif model_name == 'XGBoost':
        features_weights = dict(zip(features, model.feature_importances_))
    
    return features_weights
