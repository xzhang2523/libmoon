import torch
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def load_dataset(path, s_label):
    data = pd.read_csv( path )

    # convert categorical columns
    to_categorical = [
        'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 
        'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
    ]
    for column in to_categorical:
        data[column] = data[column].astype('category')

    data.SEX = data.SEX.replace(1, 0)   # male
    data.SEX = data.SEX.replace(2, 1)   # female

    # Scale and split
    data1 = data.copy()
    data1 = data1.drop(['default.payment.next.month', s_label], axis=1)
    data1 = pd.get_dummies(data1)

    x = StandardScaler().fit(data1).transform(data1)
    y = data['default.payment.next.month'].values
    s = data[s_label].values
    return x, y, s



class Credit(torch.utils.data.Dataset):

    def __init__(self, split, sensible_attribute='SEX'):
        assert split in ['train', 'val', 'test']

        from libmoon.util_global.constant import root_name
        path = os.path.join(root_name, 'libmoon', 'problem', 'mtl', 'mtl_data', 'credit', "credit.csv")

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


