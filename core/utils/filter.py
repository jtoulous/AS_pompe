import os
import json
import logging
import pandas as pd
from colorama import Style, Fore


def PromptColumns(df):
    numeric_columns = list(df.select_dtypes(include=['number']).columns)
    chosen_columns = []

    done = 0
    while done == 0:
        print()
        for i, column in enumerate(numeric_columns):
            print(f'{i} - {column}')
        print('\'done\' if finished')
        chosen_idx = input(Fore.GREEN + '\nChoose an index: ' + Style.RESET_ALL)
        
        if chosen_idx == 'done':
            done = 1
            break

        try:
            chosen_idx = int(chosen_idx)
            if chosen_idx < 0 or chosen_idx >= len(numeric_columns):
                raise Exception()
            chosen_columns.append(numeric_columns.pop(chosen_idx))

        except Exception as error:
            print('Bad input, try again\n')

    return chosen_columns or None



def PromptFilters(columns_to_filter):  
    if columns_to_filter is None:
        return None

    filters = {}
    for row in columns_to_filter:
        done = 0
        while done == 0:
            filter_opt = input(f'\nFilter {row}:\n 1 - Greater than\n 2 - Less than\n 3 - Equal to\n\n{Fore.GREEN}1, 2 or 3: {Style.RESET_ALL}')
            try:
                filter_opt = int(filter_opt)
                if filter_opt != 1 and filter_opt != 2 and filter_opt != 3:
                    raise Exception('bad input')
                filter_opt = 'Greater than' if filter_opt == 1 else 'Less than' if filter_opt == 2 else 'Equal to'

                filter_val = input(Fore.GREEN + '\nEnter filter Value: ' + Style.RESET_ALL)
                try: 
                    filter_val = int(filter_val)
                except Exception:
                    raise Exception()
                
                filters[row] = [filter_opt, filter_val]
                done = 1

            except Exception:
                print('Bad input, try again')
    return filters



def FilterDataframe(df, filters, save_repo):
    if filters:
        for var, (op, thresh) in filters.items():
            if op == "Greater than":
                df = df[df[var] > thresh]
            elif op == "Less than":
                df = df[df[var] < thresh]
            else:
                df = df[df[var] == thresh]

        metadata = {"filters": filters}
        metadata_path = os.path.join(save_repo, "metadata.json")
        with open(metadata_path, 'w') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
    return df



def ApplyFilters(df, filters, save_repo):
    if filters is True:
        logging.info('Filtering Data...')
        columns_to_filter = PromptColumns(df)
        if columns_to_filter:
            filters = PromptFilters(columns_to_filter)
            df = FilterDataframe(df, filters, save_repo)
    return df