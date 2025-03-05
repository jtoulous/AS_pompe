import os
import json
import yaml
import logging
import pandas as pd
from colorama import Style, Fore



def FilterDataframe(df, agent, filters, save_repo):
    logging.info('Filtering Data...')
    for var, (op, thresh) in filters.items():
        if op == "Greater than":
            df = df[df[var] > thresh]
        elif op == "Less than":
            df = df[df[var] < thresh]
        else:
            df = df[df[var] == thresh]

    with open(f'{save_repo}/{agent}/metadata.yaml', 'w') as metadata_file:
        yaml.dump(filters, metadata_file, indent=4)
    return df


def LoadFilters(df, equipment_folder):
    metadata_path = os.path.join(equipment_folder, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)
            filters_dict = metadata.get("filters", {})
            if filters_dict:
                for var, (op, thresh) in filters_dict.items():
                    if op == "Greater than":
                        df = df[df[var] > thresh]
                    elif op == "Less than":
                        df = df[df[var] < thresh]
                    else:
                        df = df[df[var] == thresh]
    return df