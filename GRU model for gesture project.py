#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split


# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[5]:


def load_processed_data(folder_path):
    files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')])
    data = []
    labels = []
    
    labels_in_files = [os.path.basename(f).split('_')[0] for f in files]
    unique_labels = sorted(set(labels_in_files))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    for file in files:
        df = pd.read_csv(file, delimiter=r'\s+', header=None)
        data.append(df.values)  # Load features (39 per frame after preprocessing)
        
        label = os.path.basename(file).split('_')[0]
        labels.append(label_map[label])
    
    return data, np.array(labels), label_map


# In[6]:


def normalize_data(train_data, val_data):
    
    all_train_data = np.concatenate(train_data, axis=0)
    mean = all_train_data.mean(axis=0)
    std = all_train_data.std(axis=0)
    
    train_data_normalized = [(seq - mean) / std for seq in train_data]
    val_data_normalized = [(seq - mean) / std for seq in val_data]
    
    return train_data_normalized, val_data_normalized


# In[7]:


class GRUModel(nn.Module):
    def __init__(self, input_size=39, hidden_size=128, num_layers=1, num_classes=30):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
       
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input, h0)
        
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        batch_size = x.size(0)
        lengths = lengths.to(device)
        last_outputs = output[torch.arange(batch_size), lengths - 1, :]
        
        out = self.fc(last_outputs)
        return out


# In[8]:


def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    lengths = torch.tensor([len(seq) for seq in sequences])
    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, lengths, torch.tensor(labels)


# In[9]:


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=50):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for sequences, lengths, labels in train_loader:
            sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if scheduler:
            scheduler.step()

        epoch_loss = running_loss / total
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")


# In[10]:


def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, lengths, labels in val_loader:
            sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    accuracy = 100 * correct / total
    avg_loss = val_loss / total
    return avg_loss, accuracy


# In[11]:


folder_path = "/Users/lianhechu/Documents/Applied AI Master Program/D7043E Advanced Data Mining/Project Data/new class/"  # Replace with your actual path
data, labels, label_map = load_processed_data(folder_path)


# In[12]:


train_sequences, val_sequences, train_labels, val_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels)


# In[13]:


train_sequences_normalized, val_sequences_normalized = normalize_data(train_sequences, val_sequences)


# In[14]:


train_dataset = list(zip(train_sequences_normalized, train_labels))
val_dataset = list(zip(val_sequences_normalized, val_labels))


# In[15]:


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# In[16]:


num_classes = len(label_map)


# In[17]:


model = GRUModel(input_size=39, hidden_size=128, num_layers=1, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50)


# In[ ]:




