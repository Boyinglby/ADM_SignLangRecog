# -*- coding: utf-8 -*-
"""
Train.
Created on Thu Apr 11 14:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/sign-language

"""


import glob
import os
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader


class PCDataset(Dataset):
    
    def __init__(self, root, transform=None, target_transform=None,
                 exclude_patterns=None):
        self.files = sorted(glob.glob(f'{root}/*.txt'))
        self.transform = transform
        self.target_transform = target_transform
        labels = sorted({os.path.basename(file).split('_')[0] \
                         for file in self.files})
        self.labelmap = {label: index for index, label in enumerate(labels)}
        if not exclude_patterns is None:
            exclude_patterns = '|'.join([pattern.strip() \
                                         for pattern in exclude_patterns])
            self.files = [file for file in self.files if not \
                          len(re.findall(exclude_patterns, file, re.I)) > 0]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        sequence = pd.read_csv(self.files[index], sep=' ').values
        sequence = sequence.reshape(len(sequence), 1, -1)
        label = os.path.basename(self.files[index]).split('_')[0]
        target = self.labelmap[label]
        if self.transform:
            sequence = self.transform(sequence)
        if self.target_transform:
            target = self.target_transform(target)
        return sequence, target


class PCDataLoader(object):
    
    def __new__(self, dataset, shuffle=True):
        return DataLoader(dataset, batch_size=1, shuffle=shuffle)


def train():
    return

if __name__ == '__main__':
    train()
