# -*- coding: utf-8 -*-
"""
Train.

"""


import copy
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import time
# import tqdm
import torch
import torch.utils.data as data
from models import RNN


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
        sequence = self._load_file_to_dataframe(index).values
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


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          epochs=1, device=None, out_dir=None):
    # set device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # show information
    print('-'*80)
    print(f'[INFO] Using device: {device.type.upper()}')
    print('-'*80)
    
    # copy model to device
    model = model.to(device)
    
    # initialize weights and accuracy of best model
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    hist = {metric: [] for metric in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}
    
    # begin iteration
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}:')
        time.sleep(0.5)
        
        
        # ---------- training phase ----------
        # set mode to training
        model.train()
        
        # initialize accumulators for loss and number of correct matches
        running_loss = 0.0
        correct_hits = 0
        
        # iterate over training data
        # for sequence, target in tqdm.tqdm(train_loader, unit=' sequence', ncols=80):
        for sequence, target in train_loader:
            # copy tensors to device
            sequence = sequence.to(device)
            target = target.to(device)
            
            # initialize hidden layer and copy to device
            hidden = model.init_hidden().to(device)
            
            # forward pass
            for inputs in sequence[0]:
                output, hidden = model(inputs, hidden)
            # print("output",output)
            loss = criterion(output, target)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # record statistics
            running_loss += loss.item() * sequence.size(0)
            predictions = torch.max(output, 1)[1]
            correct_hits += torch.sum(predictions == target).item()
        
        # update statistics for current epoch
        train_loss = running_loss / len(train_loader)

        train_acc = correct_hits / len(train_loader)
        hist['train_loss'].append(round(train_loss, 4))
        hist['train_acc'].append(round(train_acc, 4))
        
        
        # ---------- validation phase ----------
        # set mode to validation
        model.eval()
        
        # initialize accumulators for loss and number of correct matches
        running_loss = 0.0
        correct_hits = 0
        
        # iterate over validation data
        with torch.no_grad():
            for sequence, target in val_loader:
                # copy tensors to device
                sequence = sequence.to(device)
                target = target.to(device)
                
                # initialize hidden layer and copy to device
                hidden = model.init_hidden().to(device)
                
                # forward pass
                for inputs in sequence[0]:
                    output, hidden = model(inputs, hidden)
                loss = criterion(output, target)
                
                # record statistics
                running_loss += loss.item() * sequence.size(0)
                predictions = torch.max(output, 1)[1]
                correct_hits += torch.sum(predictions == target).item()
        
        # update statistics for current epoch
        val_loss = running_loss / len(val_loader)
        val_acc = correct_hits / len(val_loader)
        hist['val_loss'].append(round(val_loss, 4))
        hist['val_acc'].append(round(val_acc, 4))
        
        # print statistics for current epoch
        print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')
        time.sleep(0.5)
        
        # update and serialize model
        if val_acc > best_acc:
            best_wts = copy.deepcopy(model.state_dict())
            best_acc = val_acc
            if out_dir is not None:
                try:
                    if not os.path.isdir(out_dir):
                        os.makedirs(out_dir)
                    torch.save(model.state_dict(), f'{out_dir}/model.pt')
                    print('[INFO] Serialized model')
                except:
                    print('[INFO] Serialization failed')
        
        # update log
        if out_dir is not None:
            try:
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
                pd.DataFrame(hist).to_csv(f'{out_dir}/log.csv', index=False)
            except:
                pass
        
        print('-'*80)
    
    # return best model and history
    model.load_state_dict(best_wts)
    return model, hist

def train_loo(model, criterion, optimizer, scheduler, DATA_ROOT = './processed_data'):
    data_files = glob.glob(DATA_ROOT + '/*.txt')
    signerlist = sorted({os.path.basename(file).split('_')[1] for file in data_files})

    try:
        signerlist.remove("")
    except ValueError:
        pass
    
    for SIGNER_ID in range(len(signerlist)):     
        SAVE_PATH = f'./output/train_{SIGNER_ID}/'
        
        val_signer = [signerlist[SIGNER_ID]]
        signerlist_exlude = signerlist.copy()
        signerlist_exlude.remove(signerlist[SIGNER_ID])

        train_set = PCDataset(DATA_ROOT, exclude_patterns=val_signer)
        val_set = PCDataset(DATA_ROOT, exclude_patterns=signerlist_exlude)
        
        print('-'*80)
        print(f'[INFO] Training on {len(train_set)} samples from {len(signerlist)} signers')
        print(f'[INFO] Validating on {len(val_set)} samples from {len(val_signer)} signers')
        
        train_loader = PCDataLoader(train_set)
        val_loader = PCDataLoader(val_set)
        
        _, history = train(model, train_loader, val_loader, criterion, optimizer, scheduler,
                  epochs=10, device=None, out_dir=None)
        
        print(f'[INFO] Training finished with best validation accuracy: {max(history["val_acc"]):.4f}')
        print('-'*80)
        
        plot_history(history, SAVE_PATH, 'svg')
        
        
def plot_history(history, out_dir=None, ext='png'):
    # plot training and validation loss
    plt.figure()
    for metric, label in zip(['train_loss', 'val_loss'], ['Training', 'Validation']):
        plt.plot(range(1, len(history[metric]) + 1), history[metric], label=label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if out_dir is not None:
        try:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            plt.savefig(f'{out_dir}/loss.{ext}')
        except:
            pass
    plt.show(block=False)
    
    # plot training and validation accuracy
    plt.figure()
    for metric, label in zip(['train_acc', 'val_acc'], ['Training', 'Validation']):
        plt.plot(range(1, len(history[metric]) + 1), history[metric], label=label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if out_dir is not None:
        try:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            plt.savefig(f'{out_dir}/acc.{ext}')
        except:
            pass
    plt.show(block=False)


if __name__ == '__main__':
    
    model = RNN(60, 100, 30)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loo(model, criterion, optimizer, scheduler)
