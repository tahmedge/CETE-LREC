# utils.py

import torch
# import gensim
# from gensim.test.utils import datapath, get_tmpfile
# from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec
from torchtext import data
import spacy
import pandas as pd
from config import Config
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from imap_qa import calc_one_map
from imap_qa import calc_map1, calc_mrr1
from train import return_file_name
val = 0
class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.vocab1 = []
        self.vocab2 = []
        self.word_embeddings = {}
        self.weights = []

    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        return int(label.strip()[-1])

    def read_data(self, filename):
        with open(filename, 'r') as datafile:
            res = []
            count = 0
            for line in datafile:
                count = count + 1
                if (count == 1):
                    continue
                line = line.strip().split('\t')
                lines = []
                length = len(line)
                if (length < 3):
                    lines.append(line[0])
                    lines.append("question")
                    lines.append(line[1])
                else:
                    lines.append(line[0])
                    lines.append(line[1])
                    lines.append(line[2])

                res.append([lines[0], lines[1], float(lines[2])])

        return res

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''

        global val
        with open(filename, 'r', encoding="utf8") as datafile:
            res = []
            count=0
            #val = val
            for line in datafile:
                count=count+1
                if(count==1):
                    continue
                line = line.strip().split('\t')
                lines = []
                length = len(line)

                if (length < 3):
                    lines.append(str(val))
                    lines.append(line[0])
                    lines.append("question")
                    lines.append(line[1])
                else:
                    lines.append(str(val))
                    lines.append(line[0])
                    lines.append(line[1])
                    lines.append(line[2])
                val += 1
                #print(line)
                res.append([int(lines[0]), lines[1], lines[2], float(lines[3])])
            # return res
            # tempdata=[]
            # for line in datafile:
            #    tempdata = line.split('\t')
            #    data.append([tempdata[0],tempdata[1],tempdata[2]])
            data = res
            # print(data)
            data_index = list(map(lambda x: int(x[0]), data))
            data_text1 = list(map(lambda x: x[1], data))
            data_text2 = list(map(lambda x: x[2], data))
            data_label = list(map(lambda x: int(x[3]), data))
        full_df = pd.DataFrame({"index": data_index, "text1": data_text1, "text2": data_text2, "label": data_label})
        return full_df

    '''
    def get_pandas_df(self, filename):

        #Load the data into Pandas.DataFrame object
        #This will be used to convert data to torchtext object


        with open(filename, 'r') as datafile:
            res = []
            val = 0
            for line in datafile:
                line = line.strip().split('\t')
                lines = []
                length = len(line)
                val+=1
                if (length < 3):
                    lines.append(str(val))
                    lines.append(line[0])
                    lines.append("question")
                    lines.append(line[1])
                else:
                    lines.append(str(val))
                    lines.append(line[0])
                    lines.append(line[1])
                    lines.append(line[2])
                res.append([int(lines[0]),lines[1], lines[2], float(lines[3])])
            #return res
            #tempdata=[]
            #for line in datafile:
            #    tempdata = line.split('\t')
            #    data.append([tempdata[0],tempdata[1],tempdata[2]])
            data=res
            #print(data)
            data_index = list(map(lambda x: int(x[0]), data))
            data_text1 = list(map(lambda x: x[1], data))
            data_text2 = list(map(lambda x: x[2], data))
            data_label = list(map(lambda x: int(x[3]), data))
        full_df = pd.DataFrame({"index":data_index,"text1":data_text1, "text2":data_text2, "label":data_label})
        return full_df
    '''

    def load_data(self, train_file, test_file=None, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data

        Inputs:
            train_file (String): path to training file
            test_file (String): path to test file
            val_file (String): path to validation file
        '''

        NLP = spacy.load('en')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords

        stopWords = set(stopwords.words('english'))
        # Creating Field for data
        TEXT1 = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        TEXT2 = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        INDEX = data.Field(sequential=False, use_vocab=False)
        datafields = [("index", INDEX), ("text1", TEXT1), ("text2", TEXT2), ("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)
        # print(train_df['label'])
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)

        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)
        # print(test_data[0].__dict__.keys())
        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            print("val_file exists")
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)

        TEXT1.build_vocab(train_data, vectors="glove.6B.300d")  # fasttext.en.300d #glove.840B.300d
        TEXT2.build_vocab(train_data, vectors="glove.6B.300d")

        from torchtext import vocab
        # vec = vocab.Vectors('paragram_300.txt', '../WordVec/')
        # glove_file = ('../WordVec/paragram_300.txt')
        # tmp_file = get_tmpfile("paragram_word2vec.txt")
        # _ = glove2word2vec(glove_file, tmp_file)
        # emb = KeyedVectors.load_word2vec_format(tmp_file,encoding = "ISO-8859-1")
        # emb.save("paragram_300.txt")

        # self.weights = torch.FloatTensor(emb.vectors)

        self.vocab1 = TEXT1.vocab
        self.vocab2 = TEXT2.vocab
        self.word_embeddings1 = TEXT1.vocab.vectors
        self.word_embeddings2 = TEXT2.vocab.vectors
        #print(self.vocab1)
        # you can also write TEXT.vocab

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            # sort_key=lambda x: len(x.text),
            sort_key=lambda x: data.sort_keys(x.index),
            # sort_key=lambda x: data.interleave_keys(len(x.text1), len(x.text2)),
            repeat=False,
            shuffle=False)  # should check by making it TRUE

        self.test_iterator = data.BucketIterator(
            (test_data),
            batch_size=self.config.batch_size,

            # sort_key=lambda x: len(x.text),
            sort_key=lambda x: data.sort_keys(x.index),
            # sort_key=lambda x: data.interleave_keys(len(x.text1), len(x.text2)),
            repeat=False,
            shuffle=False)

        self.val_iterator = data.BucketIterator(
            (val_data),
            batch_size=self.config.batch_size,
            # sort_key=lambda x: len(x.text),
            sort_key=lambda x: data.sort_keys(x.index),
            # sort_key=lambda x: data.interleave_keys(len(x.text1), len(x.text2)),
            repeat=False,
            shuffle=False)
        ''''''
        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} test examples".format(len(test_data)))
        print("Loaded {} validation examples".format(len(val_data)))


def evaluate_model(model, iterator, filename):
    all_preds = []
    all_y = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(iterator):
            if torch.cuda.is_available():
                x1 = batch.text1.cuda()
                x2 = batch.text2.cuda()
                idx = batch.index.cuda()
                label = batch.label.cuda()
                # print(x2)
            else:
                x1 = batch.text1()
                x2 = batch.text2()
            y_pred = model(x1, x2, idx)
            # print(y_pred.data)
            config = Config()
            #if (config.output_size >= 2):
            #    y_pred = torch.max(y_pred.cpu().data, 1)[0]  # [0] for value, 1 for index
            # print(y_pred)
            predicted = y_pred.cpu().data.numpy()
            #print(predicted)

            # if(config.acc==1):
            #    predicted= predicted.round()
            if (config.acc == 1):
                j = 0
                for i in predicted:
                    if (predicted[j] < 0.50):
                        predicted[j] = 0.0
                    else:
                        predicted[j] = 1.0
                    j = j + 1
            # print(predicted)
            # torch.max(y_pred.cpu().data, 1)[1]+1
            all_preds.extend(predicted)
            all_y.extend(batch.label.numpy())
        # preds=np.concatenate((all_preds,all_y),axis=1)
        # print(filename)
        # print(all_y)
        # print("ok")
        config = Config()
        datasets = Dataset(config)
        t_f = datasets.read_data(filename=filename)
        #t_f="hi"
        # score = calc_map1(t_f, all_preds)
        config = Config()
        if (config.acc == 1):
            score = accuracy_score(all_y, all_preds)
            score2 = f1_score(all_y, all_preds)
            print("ACC: " + str(score * 100))
            print("F1: " + str(score2 * 100))
        else:
            # print(all_preds)
            #print(all_preds)
            score = calc_map1(t_f, all_preds)
            score2 = calc_mrr1(t_f, all_preds)
            print(filename + " MAP: " + str(score * 1))
            print(filename+" MRR: " + str(score2 * 1))
        return score,score2
'''  
def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1]
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score


def evaluate_model(model, iterator, filename, type="default"):
    all_preds = []
    all_y = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(iterator):
            if torch.cuda.is_available():
                #x1 = batch.text1.cuda()
                #x2 = batch.text2.cuda()
                x=batch.text.cuda()
                idx = batch.index.cuda()
                #print(x2)
            #else:
                #x1 = batch.text1()
                #x2 = batch.text2()
            y_pred = model(x,x,idx,type)
            #print(y_pred.data)
            config = Config()
            #if (config.output_size >= 2):
            y_pred=torch.max(y_pred.cpu().data, 1)[1] #[1] for Paraphrase detection
            #y_pred = torch.max(y_pred.cpu().data, 1)[0]
            #print(y_pred)
            #y_pred = y_pred[:, 1:]
            predicted = y_pred.cpu().data.numpy()
            #print(predicted)

            #if(config.acc==1):
            #    predicted= predicted.round()
            
            if (config.acc == 1):
                j = 0
                for i in predicted:
                    if (predicted[j] < 0.50):
                        predicted[j] = 0.0
                    else:
                        predicted[j] = 1.0
                    j = j + 1
            #print(predicted)
            #torch.max(y_pred.cpu().data, 1)[1]+1
            all_preds.extend(predicted)
            all_y.extend(batch.label.numpy())
        #preds=np.concatenate((all_preds,all_y),axis=1)
        #print(filename)
        #print(all_y)
        #print("ok")
        config = Config()
        datasets=Dataset(config)
        t_f = datasets.read_data(filename=filename)
        #score = calc_map1(t_f, all_preds)
        config=Config()
        if(config.acc==1):
            score = accuracy_score(all_y, all_preds)
            #newscore= f1_score(all_y, all_preds)
            #print("F1: "+str(newscore*100))
        else:
            #print(all_preds)
            score = calc_map1(t_f, all_preds)
            score2 = calc_mrr1(t_f, all_preds)
            print("MRR: " + str(score2 * 100))
        return score
'''