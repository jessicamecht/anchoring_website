import torch.nn as nn 
import torch
import random 
from collections import namedtuple
import torch.nn.functional as F
import functools
from collections import Counter
import operator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AnchorLSTM(nn.Module):
    '''AnchorLSTM takes in a sequence of ratings of students for the current reviewer and is 
    supposed to learn the anchor in the current hidden state'''
    def __init__(self, input_size, hidden_size, vocab_size=0, output_size=2, embedding_dim = 128):
        super(AnchorLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size= input_size, hidden_size=hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, output_size)

    
    def forward(self, x, h):#(self, x, features, h):
        '''vocab = list(map(lambda sentence: Counter(sentence.split(" ")), features))# create a dictionary
        print(vocab)
        vocab = sorted(vocab, key=vocab.get, reverse=True)
        word2idx = {word: ind for ind, word in enumerate(vocab)} 
        encoded_sentences = [word2idx[word] for word in words]
        embedded_features = self.embeddings(torch.LongTensor(encoded_sentences))
        x = torch.cat([x, embedded_features],1)'''
        x = x.reshape((1,x.shape[0], x.shape[1]))

        predictions, h = self.lstm(x, h)
        all_hidden = predictions
        predictions = torch.sigmoid(self.linear(predictions))
        return predictions, h, all_hidden
