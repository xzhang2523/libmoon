import torch
import os
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(path, s_label):

    # following the preprocessing on 
    # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    raw_data = pd.read_csv(path)

    data = raw_data[(
            (raw_data['days_b_screening_arrest'] <= 30) &
            (raw_data['days_b_screening_arrest'] >= -30) &
            (raw_data['is_recid'] != -1) &
            (raw_data['c_charge_degree'] != 'O') &
            (raw_data['score_text'] != 'N/A')
    )]

    # Only some columns are relevant
    data = data[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
                 'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

    # convert c_jail_in and c_jail_out to time_in_jail, mesured in hours
    def date_from_str(s):
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

    data['c_jail_in'] = data['c_jail_in'].apply(date_from_str)
    data['c_jail_out'] = data['c_jail_out'].apply(date_from_str)

    data['length_of_stay'] = data['c_jail_out'] - data['c_jail_in']

    # data['length_of_stay'] = data['length_of_stay'].astype('timedelta64[h]')  # modified by xz, 11.6
    data['length_of_stay'] = data['length_of_stay'].dt.days


    data = data.drop(['c_jail_in', 'c_jail_out'], axis=1)

    # encode sex
    data['sex'] = data['sex'].replace('Male', 0)
    data['sex'] = data['sex'].replace('Female', 1)

    # one-hot encode categorical variables
    data1 = data.copy()
    data1 = data1.drop(['two_year_recid', 'sex'], axis=1)
    data1 = pd.get_dummies(data1)

    x = StandardScaler().fit(data1).transform(data1)
    y = data['two_year_recid'].values
    s = data['sex'].values
    return x, y, s



class Compas(torch.utils.data.Dataset):

    def __init__(self, split, sensible_attribute='sex'):
        assert split in ['train', 'val', 'test']

        from libmoon.util_global.constant import root_name
        # folder_name = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(root_name, 'libmoon', 'problem', 'mtl', 'mtl_data', 'compas', "compas.csv")

        x, y, s = load_dataset(path, sensible_attribute)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        s = torch.from_numpy(s).long()

        # train/val/test split: 70/10/20 %
        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x, y, s, test_size=.2, random_state=1)
        x_train, x_val, y_train, y_val, s_train, s_val = train_test_split(x_train, y_train, s_train, test_size=.125, random_state=1)

        if split == 'train':
            self.x = x_train
            self.y = y_train
            self.s = s_train
        elif split == 'val':
            self.x = x_val
            self.y = y_val
            self.s = s_val
        elif split == 'test':
            self.x = x_test
            self.y = y_test
            self.s = s_test
        
        print("loaded {} instances for split {}. y positives={}, {} positives={}".format(
            len(self.y), split, sum(self.y), sensible_attribute, sum(self.s)))
        
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return dict(data=self.x[index], labels=self.y[index], sensible_attribute=self.s[index])
    
    def task_names(self):
        return None


if __name__ == "__main__":
    from torch.utils import data
    dataset = Compas(split="train")
    trainloader = data.DataLoader(dataset, batch_size=256, num_workers=0)
    for i, data in enumerate(trainloader):
        print()