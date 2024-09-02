import pickle
from libmoon.metrics import compute_indicators
import pandas as pd
import numpy as np
from libmoon.util.constant import beautiful_dict, beautiful_ind_dict, min_key_array, paper_dict
import argparse
import os

SMALL_DIGIT=3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_name', type=str, default='adult')
    parser.add_argument('--seed_num', type=int, default=1)
    parser.add_argument('--task', type=str, default='discrete')
    args = parser.parse_args()
    pickle_name_template = 'D:\\pycharm_project\\libmoon\\Output\\{}\\{}\\{}\\seed_{}\\res.pickle'

    mtd_arr = ['epo', 'mgdaub', 'random',]
    mtd_arr = ['agg_tche', 'agg_mtche', 'agg_pbi', 'agg_cosmos', 'agg_softtche',]     # For syn psl usage
    mtd_arr = ['epo', 'mgdaub', 'pmgda', 'random', 'moosvgd', 'pmtl', 'hvgrad',
               'agg_ls', 'agg_tche','agg_mtche', 'agg_pbi', 'agg_cosmos', 'agg_softtche']  # for mtl discrete usage
    task = 'task_arxiv_4'    #Task 3, synthetic psl
    if task == 'task2':
        args.problem_name = 'adult'
        args.task = 'discrete'
        mtd_arr = ['epo', 'mgdaub', 'pmgda', 'random', 'moosvgd', 'pmtl', 'hvgrad',
                   'agg_ls', 'agg_tche', 'agg_pbi', 'agg_cosmos', 'agg_softtche']  # for mtl discrete usage
    elif task == 'task3':
        args.problem_name = 'RE37'
        args.task = 'psl'
        mtd_arr = ['agg_ls','agg_tche', 'agg_mtche', 'agg_pbi', 'agg_cosmos', 'agg_softtche', 'epo', 'pmgda']  # For syn psl usage
    elif task == 'task4':
        args.seed_num=3
        args.problem_name = 'mnist'
        args.task = 'psl'
        mtd_arr = ['agg_ls','agg_tche', 'agg_mtche', 'agg_cosmos', 'agg_softtche']
    elif task == 'task_pureml':
        args.seed_num=3
        args.problem_name = 'regression'
        args.task = 'discrete'
        mtd_arr = ['agg_cosmos','agg_ls', 'agg_mtche', 'agg_tche', 'epo', 'mgdaub', 'random', 'pmtl', 'moosvgd', 'hvgrad']  # For syn psl usage
    elif task == 'task_arxiv_2':
        args.seed_num = 3
        args.problem_name = 'VLMOP2'
        args.task = 'psl'
        mtd_arr = ['agg_cosmos', 'agg_ls', 'agg_tche', 'agg_softtche', 'epo', 'pmgda']  # For syn psl usage
    elif task == 'task_arxiv_3':
        args.seed_num = 3
        args.problem_name = 'adult'
        args.task = 'discrete'
        mtd_arr = ['agg_cosmos','agg_ls', 'agg_pbi', 'agg_softmtche', 'agg_mtche', 'epo', 'mgdaub', 'random', 'pmtl', 'moosvgd', 'hvgrad']  # For syn psl usage
    elif task == 'task_arxiv_4':
        args.seed_num = 3
        args.problem_name = 'mnist'
        args.task = 'psl'
        mtd_arr = ['agg_cosmos','agg_ls', 'agg_pbi', 'agg_softtche', 'agg_tche']  # For syn psl usage
    else:
        assert False, 'Unknown task'


    mtd_mean_dict, mtd_std_dict = {}, {}

    for mtd in mtd_arr:
        indicator_arr = [0,] * args.seed_num
        for seed_idx in range(args.seed_num):
            pickle_name = pickle_name_template.format(args.task, args.problem_name, mtd, seed_idx)
            with open(pickle_name, 'rb') as f:
                res = pickle.load(f)
                objs = res['y']
                prefs = res['prefs']
                indicator_dict = compute_indicators(objs, prefs, problem_name=args.problem_name)
                indicator_arr[seed_idx] = indicator_dict
        mean_dict = {}
        std_dict = {}
        for key in indicator_arr[0].keys():
            values = [indicator_arr[seed_idx][key] for seed_idx in range(args.seed_num)]
            mean_dict[key] = np.mean(values)
            std_dict[key] = np.std(values)

        mtd_mean_dict[mtd] = mean_dict
        mtd_std_dict[mtd] = std_dict

    final_rows = []
    # Populate the rows for the final DataFrame
    for mtd in mtd_arr:
        paper_name = f' ({paper_dict[mtd]})'

        combined_row = {
            'Method': f'{beautiful_dict[mtd]}{paper_name}'
        }

        for key in mtd_mean_dict[mtd].keys():
            mean_value = mtd_mean_dict[mtd][key]
            std_value = mtd_std_dict[mtd][key]

            # Find the maximum value across all methods for the current key
            max_mean_value = max(mtd_mean_dict[mtd2][key] for mtd2 in mtd_arr)
            min_mean_value = min(mtd_mean_dict[mtd2][key] for mtd2 in mtd_arr)

            # Check if the current mean value is the largest and make it bold
            if key in min_key_array:

                if mean_value == min_mean_value:
                    combined_row[beautiful_ind_dict[key] ] = f'**{mean_value:.3f} ({std_value:.3f})**'
                else:
                    combined_row[beautiful_ind_dict[key]] = f'{mean_value:.3f} ({std_value:.3f})'
            else:
                if mean_value == max_mean_value:
                    combined_row[beautiful_ind_dict[key]] = f'**{mean_value:.3f} ({std_value:.3f})**'
                else:
                    combined_row[beautiful_ind_dict[key]] = f'{mean_value:.3f} ({std_value:.3f})'

        final_rows.append(combined_row)
    df_final = pd.DataFrame(final_rows)
    output_folder_name = os.path.join('D:\\pycharm_project\\libmoon\\script\\rebuttal', args.task, args.problem_name)
    os.makedirs(output_folder_name, exist_ok=True)
    csv_file_path = os.path.join(output_folder_name, 'indicator_mean_std.csv')
    df_final.to_csv(csv_file_path, index=False)
    print(f'Data saved to {csv_file_path}')





