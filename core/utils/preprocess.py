import logging
import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline




def PreprocessByType(df, agent_type):
    if agent_type == 'Regression':
        pass
    elif agent_type == 'Time Series':
        df = df.sort_values(by='Date')
    return df


def TimeSeriesDF(df, variable):
    df = df.sort_values(by='Date')
    preprocess_label = Pipeline([
        ('Mean', Mean(variable, label=True)),
        ('Std', Std(variable, label=True)),
        ('Max', Max(variable, label=True)),
        ('Min', Min(variable, label=True)),
        ('Rms', Rms(variable, label=True))
    ])
    df = preprocess_label.fit_transform(df)
    df = df.dropna().reset_index(drop=True)
    return df






#################################################
#####                 TEST                  ####
################################################

def MotorFeatures(dataframe):
    logging.info('Creating df for motor models...')
    df = dataframe.copy()
    preprocess = Pipeline([
        ('Motor load', MotorLoad()),
        ('Efficacite moteur', MotorEfficiency()),
        ('variation temperature', TemperatureDrift())
    ])
    df = preprocess.fit_transform(df)
    return df


def HydraulicsFeatures(dataframe):
    logging.info('Creating df for hydraulics models...')
    df = dataframe.copy()
    preprocess = Pipeline([
        ('Ratio Debit/Pression', FlowPressureRatio()),
        ('Efficacite hydraulique', HydraulicEfficiency()),
        ('Indice cavitation', CavitationIndex())
    ])
    df = preprocess.fit_transform(df)
    return df


def ElectricsFeatures(dataframe):
    logging.info('Creating df for electrics models...')
    df = dataframe.copy()
    preprocess = Pipeline([
        ('Desequilibre_tension', VoltageImbalance()),
        ('Surintensite', OvercurrentDetection()),
        ('Chauffe_anormale_module', ThermalAnomaly())
    ])
    df = preprocess.fit_transform(df)
    return df
##########################################################################





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
        X['Flow_Pressure'] = X['Flow'] / X['Pressure']
        X = X.dropna().reset_index(drop=True)
        return X



class HydraulicEfficiency(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Hydraulic efficiency'] = (X['Flow'] * X['Pressure']) / (X['Current'] * X['Voltage'])
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



class Max(TransformerMixin, BaseEstimator):
    def __init__(self, column, label=False, window=50):
        self.column = column
        self.window = window
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.label:
            X[f'Max_{self.column}'] = X[self.column].shift(1).rolling(window=self.window).max()
        else:
            X[f'Max_{self.column}'] = X[self.column].rolling(window=self.window).max()
        return X

class Min(TransformerMixin, BaseEstimator):
    def __init__(self, column, label=False, window=50):
        self.column = column
        self.window = window
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.label:
            X[f'Min_{self.column}'] = X[self.column].shift(1).rolling(window=self.window).min()
        else:
            X[f'Min_{self.column}'] = X[self.column].rolling(window=self.window).min()
        return X

class Mean(TransformerMixin, BaseEstimator):
    def __init__(self, column, label=False, window=50):
        self.column = column
        self.window = window
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.label:
            X[f'Mean_{self.column}'] = X[self.column].shift(1).rolling(window=self.window).mean()
        else:
            X[f'Mean_{self.column}'] = X[self.column].rolling(window=self.window).mean()
        return X

class Std(TransformerMixin, BaseEstimator):
    def __init__(self, column, label=False, window=50):
        self.column = column
        self.window = window
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.label:
            X[f'Std_{self.column}'] = X[self.column].shift(1).rolling(window=self.window).std()
        else:
            X[f'Std_{self.column}'] = X[self.column].rolling(window=self.window).std()
        return X

class Rms(TransformerMixin, BaseEstimator):
    def __init__(self, column, label=False, window=50):
        self.column = column
        self.window = window
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.label:
            X[f'Rms_{self.column}'] = np.sqrt((X[self.column].shift(1) ** 2).rolling(window=self.window).mean())
        else:
            X[f'Rms_{self.column}'] = np.sqrt((X[self.column] ** 2).rolling(window=self.window).mean())
        return X

class Kurt(TransformerMixin, BaseEstimator):
    def __init__(self, column, label=False, window=50):
        self.column = column
        self.window = window
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.label:
            X[f'Kurt_{self.column}'] = X[self.column].shift(1).rolling(window=self.window).kurt()
        else:
            X[f'Kurt_{self.column}'] = X[self.column].rolling(window=self.window).kurt()
        return X

class Crest(TransformerMixin, BaseEstimator):
    def __init__(self, column, label=False, window=50):
        self.column = column
        self.window = window
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.label:
            X[f'Crest_{self.column}'] = X[self.column].shift(1).rolling(window=self.window).max() / np.sqrt((X[self.column].shift(1) ** 2).rolling(window=self.window).mean())
        else:
            X[f'Crest_{self.column}'] = X[self.column].rolling(window=self.window).max() / np.sqrt((X[self.column] ** 2).rolling(window=self.window).mean())
        return X

class Form(TransformerMixin, BaseEstimator):
    def __init__(self, column, label=False, window=50):
        self.column = column
        self.window = window
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.label:
            X[f'Form_{self.column}'] = np.sqrt((X[self.column].shift(1) ** 2).rolling(window=self.window).mean()) / X[self.column].shift(1).rolling(window=self.window).mean()
        else:
            X[f'Form_{self.column}'] = np.sqrt((X[self.column] ** 2).rolling(window=self.window).mean()) / X[self.column].rolling(window=self.window).mean()
        return X

class Skew(TransformerMixin, BaseEstimator):
    def __init__(self, column, label=False, window=50):
        self.column = column
        self.window = window
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.label:
            X[f'Skew_{self.column}'] = X[self.column].shift(1).rolling(window=self.window).skew()
        else:
            X[f'Skew_{self.column}'] = X[self.column].rolling(window=self.window).skew()
        return X
