# -*- coding: utf-8 -*-
"""
Train.

"""


import copy
import glob

import os
import pandas as pd
import re
import time
import tqdm
import torch
import torch.utils.data as data
import numpy as np
from models import RNN, PACKEDRNN, EarlyStopping, BiLSTMModel, LSTMModel
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import matplotlib
# ensures that matplotlib doesn't try to open any GUI windows for plotting.
matplotlib.use('Agg')  # Set the backend before importing pyplot, matplotlib backend to a non-interactive one
import matplotlib.pyplot as plt

BATCH_SIZE = 32
NUM_CLASS = 30
HIDDEN_UNITS = 50
NUM_LAYER = 2


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
        
        
        # Load the file into a DataFrame
        self.df = pd.read_csv(self.files[index], delimiter=r'\s+', header=None)
        
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
    
    def _getmeanstd(self):
        # Initialize variables to store the sum and count of values

        dfs = []
        for index in range(len(self.files)):
            df = self._load_file_to_dataframe(index)
            dfs.append(df)
        means = [df.mean() for df in dfs]
        stds = [df.std() for df in dfs]

        # mean and std over the dataset
        means = sum(means)/len(dfs)
        stds = sum(stds)/len(dfs)
        
        return means, stds
    
    def _getminmax(self):
        # Initialize variables to store the sum and count of values

        dfs = self._load_file_to_dataframe(0)
        for index in range(1,len(self.files)):
            df = self._load_file_to_dataframe(index)
            dfs = pd.concat([df, dfs])
        mini = dfs.min() 
        maxi = dfs.max() 

        
        return mini, maxi



# Collate function to pad sequences
def collate_fn(batch):
    sequence, labels = zip(*batch)
    sequence = [torch.tensor(seq) for seq in sequence]
    lengths = torch.tensor([len(seq) for seq in sequence])
    sequence = pad_sequence(sequence, batch_first=True)
    return sequence, lengths, torch.tensor(labels)

class PCDataLoader(object):
    
    def __new__(self, dataset, shuffle=True):
        return data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4, pin_memory=True, collate_fn=collate_fn)
        # return data.DataLoader(dataset, batch_size=1, shuffle=shuffle)#, num_workers=4, pin_memory=True, collate_fn=collate_fn)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          epochs=1, device=None, out_dir='./own/bestmodel', patience=5): # updated for lstm
    # set device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # show information
    print('-'*80)
    print(f'[INFO] Using device: {device.type.upper()}')
    print('-'*80)
    # set early stop
    early_stopping = EarlyStopping(patience=patience, delta=0.01)
    # copy model to device
    model = model.to(device)
    
    # initialize weights and accuracy of best model
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    hist = {metric: [] for metric in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}
    
    # begin iteration
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}:')
        # time.sleep(0.5)
        
        # ---------- training phase ----------
        # set mode to training
        model.train()
        
        # initialize accumulators for loss and number of correct matches
        running_loss = 0.0
        correct_hits = 0
        total = 0
        
        # iterate over training data
        for sequence, lengths, target in tqdm.tqdm(train_loader, unit=' sequence', ncols=80):
            #print(lengths)
        # for sequence, target in train_loader:
            
            optimizer.zero_grad()
            # copy tensors to device
            sequence = sequence.squeeze().to(device)
            
            lengths = lengths.to('cpu')
            target = target.to(device)
            target_onehot = target
            # target_onehot = F.one_hot(target, num_classes=NUM_CLASS)
            # target_onehot = target_onehot.float()
            #-------------------------
            sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)

            output = model(sequence) # updated for lstm, need lengths as input

            
            loss = criterion(output, target_onehot)
            
            loss.backward()
            optimizer.step()
            
            # update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # record statistics
            running_loss += loss.item() # * lengths.size(0)
            predictions = torch.max(output, 1)[1]
            correct_hits += torch.sum(predictions == target).item()
            total += target.size(0)
        
        # update statistics for current epoch
        train_loss = running_loss / len(train_loader)

        train_acc = correct_hits / total
        hist['train_loss'].append(round(train_loss, 4))
        hist['train_acc'].append(round(train_acc, 4))
        
        
        # ---------- validation phase ----------
        # set mode to validation
        model.eval()
        
        # initialize accumulators for loss and number of correct matches
        running_loss = 0.0
        correct_hits = 0
        total = 0
        
        # iterate over validation data
        with torch.no_grad():
            for sequence, lengths, target in val_loader:
            # for sequence, target in train_loader:
                
                
                # copy tensors to device
                sequence = sequence.squeeze().to(device)
                lengths = lengths.to('cpu')
                target = target.to(device)
                target_onehot = target
                # target_onehot = F.one_hot(target, num_classes=NUM_CLASS)
                # target_onehot = target_onehot.float()
                #-------------------------
                sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
                

                output = model(sequence)

            
                loss = criterion(output, target_onehot)
                
                # record statistics
                
                running_loss += loss.item() #* lengths.size(0)
                predictions = torch.max(output, 1)[1]
                # print("--------------output--------------",output)
                # print("--------------prediction--------------",predictions)
                # print("--------------targets-------------",target)
                total += target.size(0)
                
                
                correct_hits += torch.sum(predictions == target).item()
        
        # update statistics for current epoch
        val_loss = running_loss / len(val_loader)
        val_acc = correct_hits / total
        hist['val_loss'].append(round(val_loss, 4))
        hist['val_acc'].append(round(val_acc, 4))
        
        print("total:",total)
        
        # print statistics for current epoch
        print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')
        
        print('-'*80)
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            torch.save(early_stopping.best_model_state, './own/bestmodel/model.pt')
            break
        
    early_stopping.load_best_model(model)
    
    # update log
    pd.DataFrame(hist).to_csv('own/log.csv', mode ='a', index=False)
    # # return best model and history
    # model.load_state_dict(best_wts)
    return model, hist

class NormalizeWithStats:
    """
    Custom transform to convert to tensor and normalize
    """
    def __init__(self, mini, maxi):
        self.min = mini
        self.max = maxi

    def __call__(self, sample):
        
        output = ((sample - self.min) / self.max - self.min)
        output = np.array(output, dtype=np.float32)
        return output
    
class StandardizeWithStats:
    """
    Custom transform to convert to tensor and normalize
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        
        output = ((sample - self.mean) / self.std)
        output = np.array(output, dtype=np.float32)
        return output
    
def train_loo(model, criterion, optimizer, scheduler, Epoch_num, DATA_ROOT = './24featuresDataset'):
    """
    leave one out cross validation
    """
    data_files = glob.glob(DATA_ROOT + '/*.txt')
    signerlist = sorted({os.path.basename(file).split('_')[1] for file in data_files})
    model_best = model
    
    val_loss_avg = 0
    val_acc_avg = 0
    
    try:
        signerlist.remove("")
    except ValueError:
        pass
    
    for SIGNER_ID in range(len(signerlist)): 
        
        val_loss_loo = 0
        val_acc_loo = 0

        SAVE_PATH = f'./output/train_{SIGNER_ID}/'
        
        val_signer = [signerlist[SIGNER_ID]]
        signerlist_exlude = signerlist.copy()
        signerlist_exlude.remove(signerlist[SIGNER_ID])

        train_set = PCDataset(DATA_ROOT, exclude_patterns=val_signer)
        val_set = PCDataset(DATA_ROOT, exclude_patterns=signerlist_exlude)
        
        # normalize in each loo iteration
        mini = train_set._getminmax()[0].tolist()
        maxi = train_set._getminmax()[1].tolist()
        transform = NormalizeWithStats(mini, maxi)
        
        train_set.transform = transform
        val_set.transform = transform

        # train 
        print('-'*80)
        print(f'[INFO] Training on {len(train_set)} samples from {len(signerlist_exlude)} signers')
        print(f'[INFO] Validating on {len(val_set)} samples from {val_signer} signers')
        
        train_loader = PCDataLoader(train_set)
        val_loader = PCDataLoader(val_set)
        if SIGNER_ID < 3:
            patience = 10
        else:
            patience = 5
        model_best, history = train(model_best, train_loader, val_loader, criterion, optimizer, scheduler,
                  epochs=Epoch_num, device=None, out_dir=None, patience=patience)
        
        print(f'[INFO] One LOO Training finished with best validation accuracy: {max(history["val_acc"]):.4f}')
        print('-'*80)
        plot_history(history, SAVE_PATH, 'svg')
        
        val_loss_loo = history['val_loss'][-1]
        val_acc_loo = history['val_acc'][-1]
        
        print(f'[INFO] One LOO Training finished with average validation accuracy: {val_acc_loo:.4f}')
        print('-'*80)
        
        val_loss_avg += val_loss_loo
        val_acc_avg += val_acc_loo
        
    val_loss_avg = val_loss_avg/10
    val_acc_avg = val_acc_avg/10
    
    print(f'10 LOO Training finished with average validation accuracy: {val_acc_avg:.4f}')
    
            
        
def plot_history(history, out_dir=None, ext='png'):
    # plot training and validation loss
    plt.figure()
    #for metric, label in zip(['train_loss', 'val_loss'], ['Training', 'Validation']):
    for metric, label in zip(['train_acc', 'val_acc'], ['Training', 'Validation']):
        plt.plot(range(1, len(history[metric]) + 1), history[metric], label=label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy') # ('Loss')
    plt.legend()
    if out_dir is not None:
        try:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            plt.savefig(f'{out_dir}/loss.{ext}')        
        except Exception as e:
            print(f"An error occurred while saving the figure: {e}")
    # plt.show(block=False)
    plt.close()  # Close the figure to free up memory
    
    # # plot training and validation accuracy
    # plt.figure()
    # for metric, label in zip(['train_acc', 'val_acc'], ['Training', 'Validation']):
    #     plt.plot(range(1, len(history[metric]) + 1), history[metric], label=label)
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # if out_dir is not None:
    #     try:
    #         if not os.path.isdir(out_dir):
    #             os.makedirs(out_dir)
    #         plt.savefig(f'{out_dir}/acc.{ext}')
    #     except:
    #         pass
    # plt.show(block=False)


    


if __name__ == '__main__':
    
    '''
    ## model RNN:  recurrent Neural Networks to capture temporal dependencies
    # model = RNN(48, 100, 4)
    # Train the model with early stopping
    
    model = PACKEDRNN(24, HIDDEN_UNITS, NUM_CLASS, NUM_LAYER)
    # model = torch.nn.RNN(24, HIDDEN_UNITS, NUM_LAYER)
    
    # criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_loo(model, criterion, optimizer, None, 50)
    '''

    ## LSTM and BiLSTM:  handle longer temporal dependencies.
    model_lstm = LSTMModel(24, HIDDEN_UNITS, NUM_CLASS, NUM_LAYER)
    model_bilstm = BiLSTMModel(24, HIDDEN_UNITS, NUM_CLASS, NUM_LAYER)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer and scheduler for LSTM
    optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=2e-3, weight_decay=2e-4)
    scheduler_lstm = torch.optim.lr_scheduler.StepLR(optimizer_lstm, step_size=10, gamma=0.1)

    # Define optimizer and scheduler for BiLSTM
    optimizer_bilstm = torch.optim.Adam(model_bilstm.parameters(), lr=2e-3, weight_decay=2e-4)
    scheduler_bilstm = torch.optim.lr_scheduler.StepLR(optimizer_bilstm, step_size=10, gamma=0.1)

    

    # Train using Leave-One-Out Cross Validation (LOO) for LSTM
    print("Training LSTM Model with Leave-One-Out Cross Validation...")
    #train_loo(model_lstm, criterion, optimizer_lstm, scheduler_lstm, epochs=50, data_root='./24featuresDataset')
    train_loo(model_lstm, criterion, optimizer_lstm, None, 50)
    '''

    # Train using Leave-One-Out Cross Validation (LOO) for BiLSTM
    print("\nTraining BiLSTM Model with Leave-One-Out Cross Validation...")
    #train_loo(model_bilstm, criterion, optimizer_bilstm, optim.lr_scheduler.StepLR, epochs=50, data_root='./24featuresDataset')
    train_loo(model_bilstm, criterion, optimizer_bilstm, None, Epoch_num=50)
    '''




