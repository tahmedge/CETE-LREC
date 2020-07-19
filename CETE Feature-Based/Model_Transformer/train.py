# train.py

#from Pytorch.Model_Transformer.utils import *
import h5py
from model import *
#from Pytorch.Model_Transformer.model_blstm import *
import numpy as np
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch
def return_file_name():
    config = Config()
    train_file = '../data/' + config.data + '_train.tsv'
    test_file = '../data/' + config.data + '_test.tsv'
    valid_file = '../data/' + config.data + '_valid.tsv'
    return train_file,test_file,valid_file

def return_emb1(k, j):
    config = Config()
    outfile = config.emb + config.data + "1"
    hf = h5py.File(outfile + ".h5", 'r')
    kk = k
    if (k % 32 != 0):
        k = k - (k % 32)
    n1 = hf.get('dataset_' + str(k))
    n1 = np.array(n1)

    if ((kk % 32) + j > 32):
        n2 = hf.get('dataset_' + str(k + 32))
        n2 = np.array(n2)

        n1 = np.concatenate((n1[kk % 32:32], n2[0:(j - (32 - (kk % 32)))]), axis=0)
        # print(n1.shape)
    else:
        n1 = n1[kk % 32:(kk % 32) + j]

    return n1


def return_emb2(k, j):
    config = Config()
    outfile = config.emb + config.data + "2"
    hf = h5py.File(outfile + ".h5", 'r')
    kk = k
    if (k % 32 != 0):
        k = k - (k % 32)
    n1 = hf.get('dataset_' + str(k))
    n1 = np.array(n1)

    if ((kk % 32) + j > 32):
        n2 = hf.get('dataset_' + str(k + 32))
        n2 = np.array(n2)
        n1 = np.concatenate((n1[kk % 32:32], n2[0:(j - (32 - (kk % 32)))]), axis=0)
    else:
        n1 = n1[kk % 32:(kk % 32) + j]
    return n1


if __name__=='__main__':
    config = Config()
    train_file = '../data/quoraTrain.txt'
    valid_file = '../data/quoraValid.txt'
    test_file = '../data/quoraTest.txt'
    train_file = '../data/msrp_train_new.txt'
    test_file = '../data/msrp_test_new.txt'
    valid_file = '../data/msrp_valid_new.txt'
    train_file = '../data/Trec_train_not_aligned.txt'
    test_file = '../data/Trec_test_not_aligned.txt'
    valid_file = '../data/Trec_valid_not_aligned.txt'
    train_file = '../data/wikitrainnewline.txt'
    test_file = '../data/wikitestnewline.txt'
    valid_file = '../data/wikidevnewline.txt'

    #if len(sys.argv) > 2:
    #    train_file = sys.argv[1]
    #if len(sys.argv) > 3:
    #    test_file = sys.argv[2]

    f = open("predicted_value.txt", "w+")
    f.close()

    train_file,test_file,valid_file=return_file_name()

    dataset = Dataset(config)
    dataset.load_data(train_file, test_file, valid_file)

    # Create Model with specified optimizer and loss function
    ##############################################################
    #model = Transformer(config, len(dataset.vocab1), len(dataset.vocab2), dataset.vocab1, dataset.vocab2, dataset.weights)
    if(config.model==0):
        model = Transformer(config, len(dataset.vocab1), len(dataset.vocab2), dataset.vocab1, dataset.vocab2, dataset.weights)
    else:
        model = TextRNN(config, len(dataset.vocab1), len(dataset.vocab2), dataset.word_embeddings1, dataset.word_embeddings2)
    if torch.cuda.is_available():
        model.cuda()
    model.train()



    #optimizer = optim.SGD(model.parameters(), lr=config.lr)
    #NLLLoss = nn.NLLLoss()
    #model.add_optimizer(optimizer)
    #model.add_loss_op(NLLLoss)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    MSELoss = nn.MSELoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(MSELoss)
    ##############################################################

    train_losses = []
    val_accuracies = []
    val_map=0
    max_val=0
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss,val_map,val_mrr= model.run_epoch(dataset.train_iterator, dataset.val_iterator, dataset.test_iterator, i) #i
        train_losses.append(train_loss)
        if(val_map>=max_val):
            max_val=val_map
            best_model = deepcopy(model)
        val_accuracies.append(val_map)

    #train_acc,tr_acc = evaluate_model(best_model, dataset.train_iterator, filename = train_file)
    #val_acc,v_acc = evaluate_model(best_model, dataset.val_iterator, filename = valid_file)
    test_acc,t_acc = evaluate_model(best_model, dataset.test_iterator, filename = test_file)


    #print('Final Training' + "\t" + str(train_acc*config.multiplyby))
    #print('Final Validation' + "\t" + str(val_acc*config.multiplyby))
    print('#Final Test MAP' + "\t" + str(test_acc*config.multiplyby))
    print('#Final Test MRR' + "\t" + str(t_acc * config.multiplyby))
    '''
    print ('Final Training Accuracy'+"\t" + str(train_acc * 100))
    print ('Final Validation Accuracy'+"\t" + str(val_acc * 100))
    print ('Final Test Accuracy'+"\t" + str(test_acc * 100))
    '''
