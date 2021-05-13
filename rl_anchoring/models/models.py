import torch.nn as nn 
import torch
import random 
from collections import namedtuple
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AnchorLSTM(nn.Module):
    '''AnchorLSTM takes in a sequence of ratings of students for the current reviewer and is 
    supposed to learn the anchor in the current hidden state'''
    def __init__(self, input_size, hidden_size, output_size=2):
        super(AnchorLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size= input_size, hidden_size=hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, output_size)

    
    def forward(self, x, h):
        x = x.reshape((1,x.shape[0], 1))
        predictions, h = self.lstm(x, h)
        all_hidden = predictions
        predictions = torch.sigmoid(self.linear(predictions))
        return predictions, h, all_hidden
