
import pickle
from libmoon.metrics import compute_indicators
import pandas as pd
import numpy as np
from libmoon.util_global.constant import beautiful_dict, beautiful_ind_dict


if __name__ == '__main__':
    problem_name = 'VLMOP1'
    seed_num=2

    pickle_name_template = 'D:\\pycharm_project\\libmoon\\Output\\discrete\\{}\\{}\\seed_{}\\res.pickle'
    mtd_arr = ['epo', 'mgdaub', 'random', 'agg_ls', 'agg_tche', 'agg_pbi', 'agg_cosmos']

    mtd_mean_dict = {}
    mtd_std_dict = {}

    for mtd in mtd_arr:
        indicator_arr = [0,] * seed_num
        for seed_idx in range(seed_num):
            pickle_name = pickle_name_template.format(problem_name, mtd, seed_idx)
            with open(pickle_name, 'rb') as f:
                res = pickle.load(f)
                objs = res['y']
                prefs = res['prefs']
                indicator_dict = compute_indicators(objs, prefs)
                indicator_arr[seed_idx] = indicator_dict
        mean_dict = {}
        std_dict = {}
        for key in indicator_arr[0].keys():
            values = [indicator_arr[seed_idx][key] for seed_idx in range(seed_num)]
            mean_dict[key] = np.mean(values)
            std_dict[key] = np.std(values)

        mtd_mean_dict[mtd] = mean_dict
        mtd_std_dict[mtd] = std_dict



    # Initialize an empty list to store the final rows
    final_rows = []

    # Populate the rows for the final DataFrame
    for mtd in mtd_arr:
        combined_row = {'Method': beautiful_dict[mtd]}

        for key in mtd_mean_dict[mtd].keys():

            mean_value = mtd_mean_dict[mtd][ key ]
            std_value = mtd_std_dict[mtd][key ]
            combined_row[beautiful_ind_dict[key] ] = f'{mean_value:.4f} ({std_value:.4f})'

        final_rows.append(combined_row)

    # Convert the list of rows to a DataFrame
    df_final = pd.DataFrame(final_rows)

    # Save the DataFrame to a CSV file
    csv_file_path = 'D:\\pycharm_project\\libmoon\\Output\\discrete\\indicator_mean_std.csv'
    df_final.to_csv(csv_file_path, index=False)

    print(f'Data saved to {csv_file_path}')





