import pandas as pd
import argparse as ap
from utils.feature_engineering import MotorFeatures, HydraulicsFeatures, ElectricsFeatures



def Parsing():
    parser = ap.ArgumentParser()
    parser.add_argument('datafile', type=str, help='csv df')
    parser.add_argument('output', type=str, help='output csv')
    return parser.parse_args()

if __name__ == '__main__':
    args = Parsing()
    df = pd.read_csv(args.datafile, sep=';')

    df = MotorFeatures(df)
    df = HydraulicsFeatures(df)
    df = ElectricsFeatures(df)

    df.to_csv(args.output, sep=';', index=False)