# train_utils.py

import torch
from torch import nn
from torch.autograd import Variable
import copy
import math
#from Pytorch.Model_Transformer.utils import Dataset as utils
#from Pytorch.Model_Transformer.config import Config

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
    '''
    Usual Embedding layer with weights multiplied by sqrt(d_model)
    '''
    def __init__(self, d_model, vocab, v, u):
        super(Embeddings, self).__init__()
        #self.lut = nn.Embedding(vocab, d_model)
        #v.vectors=

        #config=Config()
        #u = utils(config)
        #print(u)
        self.lut = nn.Embedding(vocab, d_model).from_pretrained(v.vectors)
            #(v.vectors)
        #self.lut.weight.data.copy_(vocab.vectors)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe[:, 1::2] = torch.cos(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))#torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)