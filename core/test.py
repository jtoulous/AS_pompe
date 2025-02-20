import pandas as pd
from sklearn.pipeline import Pipeline

from utils.feature_engineering import MotorFeatures, HydraulicsFeatures, ElectricsFeatures
from utils.feature_engineering import Max, Min, Mean, Std, Skew, Kurt, Rms, Crest, Form




if __name__ == '__main__':
    try:
        df_master = pd.read_csv('data/anormal.csv', sep=';')
        df_motor = MotorFeatures(df_master)
        df_hydro = HydraulicsFeatures(df_master)
        df_elec = ElectricsFeatures(df_master)


        for col in df_master.select_dtypes(include=['number']).columns:
            df_master = Max(col).transform(df_master)
            df_master = Min(col).transform(df_master)
            df_master = Mean(col).transform(df_master)
            df_master = Std(col).transform(df_master)
            df_master = Rms(col).transform(df_master)
            df_master = Kurt(col).transform(df_master)
            df_master = Crest(col).transform(df_master)
            df_master = Form(col).transform(df_master)
            df_master = Skew(col).transform(df_master)


        for col in df_motor.select_dtypes(include=['number']).columns:
            df_motor = Max(col).transform(df_motor)
            df_motor = Min(col).transform(df_motor)
            df_motor = Mean(col).transform(df_motor)
            df_motor = Std(col).transform(df_motor)
            df_motor = Rms(col).transform(df_motor)
            df_motor = Kurt(col).transform(df_motor)
            df_motor = Crest(col).transform(df_motor)
            df_motor = Form(col).transform(df_motor)
            df_motor = Skew(col).transform(df_motor)

        for col in df_hydro.select_dtypes(include=['number']).columns:
            df_hydro = Max(col).transform(df_hydro)
            df_hydro = Min(col).transform(df_hydro)
            df_hydro = Mean(col).transform(df_hydro)
            df_hydro = Std(col).transform(df_hydro)
            df_hydro = Rms(col).transform(df_hydro)
            df_hydro = Kurt(col).transform(df_hydro)
            df_hydro = Crest(col).transform(df_hydro)
            df_hydro = Form(col).transform(df_hydro)
            df_hydro = Skew(col).transform(df_hydro)

        for col in df_elec.select_dtypes(include=['number']).columns:
            df_elec = Max(col).transform(df_elec)
            df_elec = Min(col).transform(df_elec)
            df_elec = Mean(col).transform(df_elec)
            df_elec = Std(col).transform(df_elec)
            df_elec = Rms(col).transform(df_elec)
            df_elec = Kurt(col).transform(df_elec)
            df_elec = Crest(col).transform(df_elec)
            df_elec = Form(col).transform(df_elec)
            df_elec = Skew(col).transform(df_elec)


        df_master = df_master[50:]
        df_motor = df_motor[50:]
        df_hydro = df_hydro[50:]
        df_elec = df_elec[50:]

        df_master.to_csv('data/a_master.csv', sep=';', index=False)
        df_motor.to_csv('data/a_motor.csv', sep=';', index=False)
        df_hydro.to_csv('data/a_hydro.csv', sep=';', index=False)
        df_elec.to_csv('data/a_elec.csv', sep=';', index=False)



    except Exception as error:
        print(error)