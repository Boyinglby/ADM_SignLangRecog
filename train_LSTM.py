import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import pandas as pd
import numpy as np
import glob
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

BATCH_SIZE = 32
NUM_CLASS = 30
HIDDEN_UNITS = 50
NUM_LAYER = 2


# Dataset definition
class PCDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, exclude_patterns=None):
        self.files = sorted(glob.glob(f'{root}/*.txt'))
        self.transform = transform
        self.target_transform = target_transform
        labels = sorted({os.path.basename(file).split('_')[0] for file in self.files})
        self.labelmap = {label: index for index, label in enumerate(labels)}
        if exclude_patterns is not None:
            exclude_patterns = '|'.join([pattern.strip() for pattern in exclude_patterns])
            self.files = [file for file in self.files if not len(re.findall(exclude_patterns, file, re.I)) > 0]

    def __len__(self):
        return len(self.files)

    def _load_file_to_dataframe(self, index):
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

    def _getminmax(self):
        dfs = self._load_file_to_dataframe(0)
        for index in range(1, len(self.files)):
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


# DataLoader
class PCDataLoader(object):
    def __new__(self, dataset, shuffle=True):
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4, pin_memory=True,
                          collate_fn=collate_fn)


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(hidden[-1])
        return output


# BiLSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.bilstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))  # Concatenate both directions
        return output


# Training function
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=1, device=None, patience=5):
    # set device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = model.to(device)
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}:')
        model.train()
        running_loss = 0.0
        correct_hits = 0
        total = 0

        for sequence, lengths, target in tqdm.tqdm(train_loader, unit=' sequence', ncols=80):
            optimizer.zero_grad()
            sequence = sequence.squeeze().to(device)
            lengths = lengths.to('cpu')
            target = target.to(device)

            output = model(sequence, lengths)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = torch.max(output, 1)[1]
            correct_hits += torch.sum(predictions == target).item()
            total += target.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_hits / total
        hist['train_loss'].append(round(train_loss, 4))
        hist['train_acc'].append(round(train_acc, 4))

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct_hits = 0
        total = 0

        with torch.no_grad():
            for sequence, lengths, target in val_loader:
                sequence = sequence.squeeze().to(device)
                lengths = lengths.to('cpu')
                target = target.to(device)

                output = model(sequence, lengths)
                loss = criterion(output, target)

                running_loss += loss.item()
                predictions = torch.max(output, 1)[1]
                total += target.size(0)
                correct_hits += torch.sum(predictions == target).item()

        val_loss = running_loss / len(val_loader)
        val_acc = correct_hits / total
        hist['val_loss'].append(round(val_loss, 4))
        hist['val_acc'].append(round(val_acc, 4))

        print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')

    return model, hist


# Function to train using Leave-One-Out Cross Validation (LOO)
def train_loo(model_class, criterion, optimizer_class, scheduler_class, epochs, data_root='./24featuresDataset'):
    data_files = glob.glob(data_root + '/*.txt')
    signerlist = sorted({os.path.basename(file).split('_')[1] for file in data_files})
    val_loss_avg = 0
    val_acc_avg = 0

    for SIGNER_ID in range(len(signerlist)):
        val_signer = [signerlist[SIGNER_ID]]
        signerlist_exlude = signerlist.copy()
        signerlist_exlude.remove(signerlist[SIGNER_ID])

        train_set = PCDataset(data_root, exclude_patterns=val_signer)
        val_set = PCDataset(data_root, exclude_patterns=signerlist_exlude)

        mini = train_set._getminmax()[0].tolist()
        maxi = train_set._getminmax()[1].tolist()
        transform = NormalizeWithStats(mini, maxi)

        train_set.transform = transform
        val_set.transform = transform

        train_loader = PCDataLoader(train_set)
        val_loader = PCDataLoader(val_set)

        model = model_class(24, HIDDEN_UNITS, NUM_CLASS, NUM_LAYER)
        optimizer = optimizer_class(model.parameters(), lr=2e-3, weight_decay=2e-4)
        scheduler = scheduler_class(optimizer, step_size=10, gamma=0.1)

        print(f'Training signer {SIGNER_ID+1}...')

        trained_model, history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs)

        val_loss_avg += history['val_loss'][-1]
        val_acc_avg += history['val_acc'][-1]

    val_loss_avg /= len(signerlist)
    val_acc_avg /= len(signerlist)

    print(f'Final Average Validation Loss: {val_loss_avg:.4f}')
    print(f'Final Average Validation Accuracy: {val_acc_avg:.4f}')


# Utility classes for normalization
class NormalizeWithStats:
    def __init__(self, mini, maxi):
        self.min = mini
        self.max = maxi

    def __call__(self, sample):
        output = (sample - self.min) / (self.max - self.min)
        output = np.array(output, dtype=np.float32)
        return output


# Main Function
if __name__ == '__main__':
    # Define model
    model_lstm = LSTMModel(24, HIDDEN_UNITS, NUM_CLASS, NUM_LAYER)
    model_bilstm = BiLSTMModel(24, HIDDEN_UNITS, NUM_CLASS, NUM_LAYER)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer and scheduler for LSTM
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=2e-3, weight_decay=2e-4)
    scheduler_lstm = optim.lr_scheduler.StepLR(optimizer_lstm, step_size=10, gamma=0.1)

    # Define optimizer and scheduler for BiLSTM
    optimizer_bilstm = optim.Adam(model_bilstm.parameters(), lr=2e-3, weight_decay=2e-4)
    scheduler_bilstm = optim.lr_scheduler.StepLR(optimizer_bilstm, step_size=10, gamma=0.1)

    # Train using Leave-One-Out Cross Validation (LOO) for LSTM
    print("Training LSTM Model with Leave-One-Out Cross Validation...")
    train_loo(LSTMModel, criterion, optim.Adam, optim.lr_scheduler.StepLR, epochs=50, data_root='./24featuresDataset')

    # Train using Leave-One-Out Cross Validation (LOO) for BiLSTM
    print("\nTraining BiLSTM Model with Leave-One-Out Cross Validation...")
    train_loo(BiLSTMModel, criterion, optim.Adam, optim.lr_scheduler.StepLR, epochs=50, data_root='./24featuresDataset')
    
    print("Training Complete for LSTM and BiLSTM Models!")

   