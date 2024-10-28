
import torch.nn as nn
import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence

class RNN(torch.nn.Module):
    
    def __init__(self, input_units, hidden_units, output_units):
        super(RNN, self).__init__()
        self.hidden_units = hidden_units
        self.i2h = torch.nn.Linear(input_units + hidden_units, hidden_units)
        self.i2o = torch.nn.Linear(input_units + hidden_units, output_units)
        self.out = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, hidden):

        concat = torch.cat((inputs, hidden), dim=1)
        hidden = torch.tanh(self.i2h(concat))  # Apply tanh activation
        # hidden = self.i2h(concat)
        output = self.i2o(concat)
        # output = self.out(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.randn(1, self.hidden_units)
    
class PACKEDRNN(torch.nn.Module):
    
    def __init__(self, input_units, hidden_units, output_units, num_layer):
        super(PACKEDRNN, self).__init__()
        self.hidden_units = hidden_units
        self.rnn = torch.nn.RNN(input_units, hidden_units, num_layer)
        self.fc = torch.nn.Linear(hidden_units, output_units)
        self.out = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, packed_seq):
        # print("hidden shape", hidden.shape)
        # print("inputs shape", inputs.shape)


        output, hidden = self.rnn(packed_seq)
        output, _ = pad_packed_sequence(output, batch_first=True)

        output = self.fc(output)
        output = self.out(output)
        #print(output.shape)
        return output[:,-1,:]
    
    def init_hidden(self):
        return torch.randn(1, self.hidden_units)
    
class VanillaRNN(torch.nn.Module):
    
    def __init__(self, input_units, hidden_units, output_units):
        super(VanillaRNN, self).__init__()
        self.hidden_units = hidden_units
        self.i2h = torch.nn.Linear(input_units, hidden_units)
        self.h2h = torch.nn.Linear(hidden_units, hidden_units)
        self.h2o = torch.nn.Linear(hidden_units, output_units)
        self.out = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, hidden):

        inputs = self.i2h(inputs)
        hidden = torch.tanh(self.h2h(inputs+hidden))  # Apply tanh activation
        # hidden = self.i2h(concat)
        output = self.h2o(hidden)
        output = self.out(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.randn(1, self.hidden_units)
    
# Implement Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, packed_seq):

        packed_output, (hidden, _) = self.lstm(packed_seq)
        
        output = self.fc(hidden[-1])
        output = self.softmax(output)
        return output
# output, _ = pad_packed_sequence(packed_output, batch_first=True)

# BiLSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, packed_seq):
        
        packed_output, (hidden, _) = self.bilstm(packed_seq)

        output = self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))  # Concatenate both directions
        output = self.softmax(output)
        return output
# output, _ = pad_packed_sequence(packed_output, batch_first=True)    


    
