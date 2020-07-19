# model.py

import torch
from copy import deepcopy
from torch import nn
import numpy as np
from torch.nn import functional as F
from Pytorch.Model_Transformer.utils import *
from Pytorch.Model_Transformer.train import return_emb1,return_emb2


class TextRNN(nn.Module):
    def __init__(self, config, vocab_size1,vocab_size2, word_embeddings1, word_embeddings2):
        super(TextRNN, self).__init__()
        self.config = config

        # Embedding Layer
        self.embeddings1 = nn.Embedding(vocab_size1, self.config.embed_size)
        self.embeddings1.weight = nn.Parameter(word_embeddings1, requires_grad=False)

        self.embeddings2 = nn.Embedding(vocab_size2, self.config.embed_size)
        self.embeddings2.weight = nn.Parameter(word_embeddings2, requires_grad=False)

        self.lstm1 = nn.LSTM(input_size=self.config.embed_size,
                            hidden_size=self.config.hidden_size,
                            num_layers=self.config.hidden_layers,
                            dropout=self.config.dropout_keep,
                            bidirectional=self.config.bidirectional)
        self.lstm2 = nn.LSTM(input_size=self.config.embed_size,
                             hidden_size=self.config.hidden_size,
                             num_layers=self.config.hidden_layers,
                             dropout=self.config.dropout_keep,
                             bidirectional=self.config.bidirectional)

        self.dropout = nn.Dropout(self.config.dropout_keep)

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.hidden_size * self.config.hidden_layers * (1 + self.config.bidirectional),
            self.config.output_size
        )

        # Softmax non-linearity
        self.softmax = nn.Softmax()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-1)
    def forward(self, x, y, idx):
        # x.shape = (max_sen_len, batch_size)
        #embedded_e_sent1= self.embeddings1(x)
        #embedded_sent2 = self.embeddings2(y)

        idx = idx.data.cpu().numpy()
        embedded_sents1 = return_emb1(int(idx[0]), idx.size)
        embedded_sents2 = return_emb2(int(idx[0]), idx.size)

        #embedded_sents1 = torch.from_numpy(embedded_sents1).float().cuda()
        #embedded_sents2 = torch.from_numpy(embedded_sents2).float().cuda()

        #print(embedded_e_sent1.size())
        #print(embedded_sents2.size())
        embedded_sents1 = embedded_sents1.swapaxes(1, 0)#, 2)
        embedded_sents2 = embedded_sents2.swapaxes(1, 0)#, 2)

        embedded_sents1 = torch.from_numpy(embedded_sents1).float().cuda()
        embedded_sents2 = torch.from_numpy(embedded_sents2).float().cuda()
        #print(embedded_sents2.shape)
        # embedded_sent.shape = (max_sen_len=20, batch_size=64,embed_size=300)

        lstm_out1, (h_1n, c1_n) = self.lstm1(embedded_sents1)
        lstm_out2, (h_2n, c2_n) = self.lstm2(embedded_sents2)
        final_feature_map1 = self.dropout(h_1n)  # shape=(num_layers * num_directions, 64, hidden_size)
        final_feature_map2 = self.dropout(h_2n)

        #final_feature_map1=lstm_out1
        #final_feature_map2=lstm_out2
        #print(lstm_out1.size())
        # Convert input to (64, hidden_size * hidden_layers * num_directions) for linear layer
        final_feature_map1 = torch.cat([final_feature_map1[i, :, :] for i in range(final_feature_map1.shape[0])], dim=1)
        final_feature_map2 = torch.cat([final_feature_map2[i, :, :] for i in range(final_feature_map2.shape[0])], dim=1)
        #print(final_feature_map1.size())
        #final_feature_map1=torch.mean(final_feature_map1,1)
        #final_feature_map2=torch.mean(final_feature_map2,1)
        #final_feature_map1 = torch.mean(final_feature_map1,0)
        #final_feature_map2 = torch.mean(final_feature_map2,0)
        #print(final_feature_map1)
        #print(final_feature_map2)
        #print(final_feature_map1.shape)
        #print(output)
        final_out1 = self.fc(final_feature_map1)
        final_out2 = self.fc(final_feature_map2)
        #final_out1 = final_out1.squeeze()
        #final_out2 = final_out2.squeeze()
        #print(final_out1.shape)
        output = self.cos(final_out1 , final_out2 )
        #return self.softmax(final_out)
        return output
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, test_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()
        self.train()
        for i, batch in enumerate(train_iterator):

            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x1 = batch.text1.cuda()
                x2 = batch.text2.cuda()
                idx = (batch.index).type(torch.cuda.FloatTensor)
                y = (batch.label).type(torch.cuda.FloatTensor)
            else:
                #x = batch.text1
                y = (batch.label).type(torch.FloatTensor)
            y_pred = self.__call__(x1,x2,idx)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                #val_accuracy = evaluate_model(self, val_iterator)
                #print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()
        #trainfilename, testfilename,
        validfilename = '../data/Trec_valid_not_aligned.txt'
        val_map,val_mrr=evaluate_model(self,val_iterator,validfilename)
        return train_losses, val_map,val_mrr