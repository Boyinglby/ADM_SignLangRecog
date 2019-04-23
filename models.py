# -*- coding: utf-8 -*-
"""
Models.
Created on Mon Apr 22 14:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/sign-language

"""


import torch


class RNN(torch.nn.Module):
    
    def __init__(self, input_units, hidden_units, output_units):
        super(RNN, self).__init__()
        self.hidden_units = hidden_units
        self.i2h = torch.nn.Linear(input_units + hidden_units, hidden_units)
        self.i2o = torch.nn.Linear(input_units + hidden_units, output_units)
        self.out = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, hidden):
        concat = torch.cat((inputs, hidden), 1)
        hidden = self.i2h(concat)
        output = self.i2o(concat)
        output = self.out(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_units)
