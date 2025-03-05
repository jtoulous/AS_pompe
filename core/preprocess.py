import pandas as pd
import argparse as ap

def Parsing():
    parser = ap.ArgumentParser()
    parser.add_argument('datafile', type=str, help='csv df')
    parser.add_argument('output', type=str, help='output csv')
    return parser.parse_args()

def generate_features(df):
    # Features pour le moteur
    df['motor_load'] = df['raw_power'] / df['instantaneous_flow']
    df['motor_efficiency'] = df['instantaneous_flow'] / df['raw_power']
    df['acceleration'] = df['speed'].diff().fillna(0)

    # Features pour l'hydraulique
    df['flow_resistance'] = df['headloss'] / df['instantaneous_flow']
    df['hydraulic_efficiency'] = df['instantaneous_flow'] / df['integrated_flow']
    df['flow_stability'] = df['instantaneous_flow'].rolling(5).std().fillna(0)

    # Features pour l'électrique
    df['power_stability'] = df['raw_power'].rolling(5).std().fillna(0)
    df['power_speed_ratio'] = df['raw_power'] / df['speed']
    df['energy_loss'] = df['headloss'] / df['raw_power']

    # Suppression des NaN avant de retourner le dataframe
    df = df.dropna()

    return df

if __name__ == '__main__':
    args = Parsing()
    df = pd.read_csv(args.datafile, sep=';')

    # Génération des nouvelles features
    df = generate_features(df)

    # Sauvegarde le fichier avec les nouvelles features
    df.to_csv(args.output, sep=';', index=False)
