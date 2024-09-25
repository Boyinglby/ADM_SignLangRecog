import copy
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import torch.utils.data as data


class PCDataset(data.Dataset):
    
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
    
    def _load_file_to_dataframe(self, index):
        # Define column names for 20 points with 3D coordinates (p1_x, p1_y, p1_z, ..., p20_x, p20_y, p20_z)
        coordinates = ['x', 'y', 'z']
        column_names = [f'p{point}_{coord}' for point in range(1, 21) for coord in coordinates]
        
        # Load the file into a DataFrame
        self.df = pd.read_csv(self.files[index], delimiter=r'\s+', header=None, names=column_names)
        
        return self.df
    
    def __getitem__(self, index):
        sequence = self._load_file_to_dataframe(self, index).values
        sequence = sequence.reshape(len(sequence), 1, -1).astype('float32')
        label = os.path.basename(self.files[index]).split('_')[0]
        target = self.labelmap[label]
        if self.transform:
            sequence = self.transform(sequence)
        if self.target_transform:
            target = self.target_transform(target)
        return sequence, target


class PCDataLoader(object):
    
    def __new__(self, dataset, shuffle=True):
        return data.DataLoader(dataset, batch_size=1, shuffle=shuffle)

    

if __name__ == '__main__':
    DATA_ROOT = './processed_data'
    SIGNER_ID = 0
    NUM_EPOCH = 1
    SAVE_PATH = f'./output/train_{SIGNER_ID}/'
    
    data_files = glob.glob(DATA_ROOT + '/*.txt')
    signerlist = sorted({os.path.basename(file).split('_')[1] for file in data_files})
    signerlist.remove("")
    val_signer = [signerlist.pop(SIGNER_ID)]
    
    train_set = PCDataset(DATA_ROOT, exclude_patterns=val_signer)
    val_set = PCDataset(DATA_ROOT, exclude_patterns=signerlist)
    
    print('-'*80)
    print(f'[INFO] Training on {len(train_set)} samples from {len(signerlist)} signers')
    print(f'[INFO] Validating on {len(val_set)} samples from {len(val_signer)} signers')
    
    train_loader = PCDataLoader(train_set)
    val_loader = PCDataLoader(val_set)