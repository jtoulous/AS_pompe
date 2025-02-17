import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline





## ===================== Features pour la Maintenance Prédictive =====================
## Features de base (les vibes pures de notre système)
#features_base = [
#    "Debit",                    # Débit
#    "Pression",                 # Pression
#    "Puissance",                # Puissance
#    "Couple",                   # Couple
#    "Frequence",                # Fréquence
#    "Tension",                  # Tension (mesure principale)
#    "Courant",                  # Courant
#    "Intensite_courant_thermique",  # Intensité du courant thermique
#    "Tension_entree",           # Tension d'entrée
#    "Temperature_module",       # Température du module
#    "Tension_pile_memoire"      # Tension pile mémoire
#]
#
## ======= Premier Niveau : Détection d'Anomalie Globale =======
## On utilise ici les mesures brutes et quelques stats pour capter toute vibration anormale.
#features_global = {
#    "stats": {
#        # Calculs statistiques sur une fenêtre temporelle (moyenne, écart-type, variation)
#        "moyenne": {var: f"mean({var})" for var in features_base},
#        "ecart_type": {var: f"std({var})" for var in features_base},
#        "variation": {var: f"delta({var})" for var in features_base}
#    },
#    "ratios": {
#        "Puissance_sur_Couple": "Puissance / Couple",  # Indique la charge moteur approximative
#        "Debit_sur_Pression": "Debit / Pression",       # Indicateur de performance hydraulique
#        "Courant_sur_Tension": "Courant / Tension",       # Équilibre électrique
#        "Temperature_sur_Intensite": "Temperature_module / Intensite_courant_thermique"
#    }
#}
#
## ======= Deuxième Niveau : Localisation de l'Anomalie =======
## On cible les composants individuels pour isoler l’anomalie, toujours en s’appuyant sur nos bases.
#
## ----- Moteur -----
#features_moteur = {
#    "Charge_moteur": "Couple / Puissance",              # Charge appliquée au moteur
#    "Efficacite_moteur": "Puissance / (Courant * Tension)",  # Rendement énergétique du moteur
#    "Derive_frequence": "Frequence - mean_reference"    # Déviation par rapport à une fréquence de référence
#}
#
## ----- Hydraulique -----
#features_hydraulique = {
#    "Ratio_Debit_Pression": "Debit / Pression",         # Ratio indiquant l’efficacité hydraulique
#    "Efficacite_hydraulique": "(Debit * Pression) / Puissance",  # Puissance hydraulique effective
#    "Indice_cavitation": "Pression - pression_reference" # Différence par rapport à une pression d'aspiration normale
#}
#
## ----- Electrique -----
#features_electrique = {
#    # Pour 'Desequilibre_tension', on suppose que Tension_phases est accessible (mesures par phase)
#    "Desequilibre_tension": "max(Tension_phases) - min(Tension_phases)",
#    "Surintensite": "Courant / courant_nominal",       # Ratio par rapport au seuil nominal de courant
#    "Chauffe_anormale_module": "Temperature_module - k * Intensite_courant_thermique"  # Excès de chaleur détecté
#}
#
## Regroupement des features élaborées pour nos modèles
#features_engineered = {
#    "global": features_global,
#    "moteur": features_moteur,
#    "hydraulique": features_hydraulique,
#    "electrique": features_electrique
#}





def MotorFeatures(df):
    preprocess = Pipeline([
        ('Charge moteur', ChargeMoteur()),
        ('Efficacite moteur', EfficaciteMoteur()),
        ('Derive frequence', DeriveFrequence())
    ])
    df = preprocess.fit_transform(df)
    
    wanted_features = [
        'Charge Moteur',
        'Efficacite Moteur',
        'Derive Frequence'
    ]
    return df[wanted_features]


def HydraulicsFeatures(df):
    preprocess = Pipeline([
        ('Ratio Debit/Pression', DebitPression()),
        ('Efficacite hydraulique', EfficaciteHydrau()),
        ('Indice cavitation', IndiceCavitation())
    ])
    df = preprocess.fit_transform(df)
    
    wanted_features = [
        'Debit/Pression',
        'Efficacite hydraulique',
        'Indice cavitation'
    ]
    return df[wanted_features]



def ElectricsFeatures(df):
    preprocess = Pipeline([
        ('Desequilibre_tension', DesequilibreTension()),
        ('Surintensite', Surintensite()),
        ('Chauffe_anormale_module', ChauffeModule())
    ])
    df = preprocess.fit_transform(df)
    
    wanted_features = [
        'Desequilibre tension',
        'Surintensite',
        'Chauffe module'
    ]
    return df[wanted_features]