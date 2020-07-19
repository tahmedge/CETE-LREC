# config.py
class Config(object):
    N = 1 #6 in Transformer Paper
    d_model = 768 #512 in Transformer Paper
    d_ff = 768 #2048 in Transformer Paper
    embed_size=300
    h = 6
    dropout = 0.10
    output_size = 1
    lr = 0.00005
    max_epochs = 1
    batch_size = 32
    max_sen_len = 50
    acc=0 #1 for msrp, 0 for trec
    multiplyby=100
    hidden_layers = 2
    hidden_size = 768
    bidirectional = True
    dropout_keep = 0.1
    model=0 #0 for transformer, 1 for blstm
    #emb="../data/bert"
    EmbeddingType = "Contextualized" #Otherwise, EmbeddingType="Contextualized" or "Baseline"
    emb="E:/H5PY Data/BERT_FILE/bert"
    data="yahoo"
    #data = "semeval2017"
    #data="wiki"
    #data = "trecr"
    #data = "trecr"
    # data = "semeval2016"
    # data = "yahoo"



    # trecc 0.405 0.476
    # wiki 0.566 0.571
    # YAHOO 0.436, 0.436
    # semeval2016 0.604, 0.670
    # semeval2017 0.700 0.757