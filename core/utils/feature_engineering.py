import logging
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline





def MotorFeatures(dataframe):
    logging.info('Creating df for motor models...')
    df = dataframe.copy()
    preprocess = Pipeline([
        ('Motor load', MotorLoad()),
        ('Efficacite moteur', MotorEfficiency()),
        ('variation temperature', TemperatureDrift())
    ])
    df = preprocess.fit_transform(df)
    
    wanted_features = [
        'Date',
        'Motor load',
        'Motor efficiency',
        'Temperature drift'
    ]
    return df[wanted_features]


def HydraulicsFeatures(dataframe):
    logging.info('Creating df for hydraulics models...')
    df = dataframe.copy()
    preprocess = Pipeline([
        ('Ratio Debit/Pression', FlowPressureRatio()),
        ('Efficacite hydraulique', HydraulicEfficiency()),
        ('Indice cavitation', CavitationIndex())
    ])
    df = preprocess.fit_transform(df)
    
    wanted_features = [
        'Date',
        'Flow_Pressure',
        'Hydraulic efficiency',
        'Cavitation index'
    ]
    return df[wanted_features]



def ElectricsFeatures(dataframe):
    logging.info('Creating df for electrics models...')
    df = dataframe.copy()
    preprocess = Pipeline([
        ('Desequilibre_tension', VoltageImbalance()),
        ('Surintensite', OvercurrentDetection()),
        ('Chauffe_anormale_module', ThermalAnomaly())
    ])
    df = preprocess.fit_transform(df)
    
    wanted_features = [
        'Date',
        'Voltage imbalance',
        'Overcurrent detection',
        'Thermal anomaly'
    ]
    return df[wanted_features]






##################################################################
######             Transformers for pipeline                 #####
##################################################################
                             #####
                         #############
                           #########
                              ###
                               #


class MotorLoad(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Motor load'] = X['Current'] * X['Voltage'] / X['Temperature']
        X = X.dropna().reset_index(drop=True)
        return X



class MotorEfficiency(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Motor efficiency'] = (X['Current'] * X['Voltage']) / X['Pressure']
        X = X.dropna().reset_index(drop=True)
        return X



class TemperatureDrift(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Temperature drift'] = X['Temperature'] - X['Temperature'].rolling(window=10).median()
        X = X.dropna().reset_index(drop=True)
        return X



class FlowPressureRatio(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Flow_Pressure'] = X['Volume Flow RateRMS'] / X['Pressure']
        X = X.dropna().reset_index(drop=True)
        return X



class HydraulicEfficiency(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Hydraulic efficiency'] = (X['Volume Flow RateRMS'] * X['Pressure']) / (X['Current'] * X['Voltage'])
        X = X.dropna().reset_index(drop=True)
        return X


class CavitationIndex(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Cavitation index'] = X['Pressure'] - X['Pressure'].rolling(window=10).mean()
        X = X.dropna().reset_index(drop=True)
        return X



class VoltageImbalance(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Voltage imbalance'] = X['Voltage'] - X['Voltage'].rolling(window=10).mean() 
        X = X.dropna().reset_index(drop=True)
        return X


class OvercurrentDetection(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Overcurrent detection'] = X['Current'] / X['Current'].mean()
        X = X.dropna().reset_index(drop=True)
        return X



class ThermalAnomaly(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Thermal anomaly'] = X['Temperature'] / X['Current']
        X = X.dropna().reset_index(drop=True)
        return X
