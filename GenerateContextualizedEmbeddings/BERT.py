import numpy as np
import h5py
import mxnet as mx
from bert_embedding import BertEmbedding
#https://github.com/imgarylai/bert-embedding
f1= open("bertS16size1.txt","w+")
f2= open("bertS16size2.txt","w+")
def read_data(filename):
    with open(filename, 'r',encoding="utf8") as datafile:
        res = []
        for line in datafile:
            line = line.strip().split('\t')

            lines = line
            length = len(line)
            track=0
            for x in lines[0]:
                if(x!=' '):
                    track+=1
            if(track==0):
                print(lines)
                lines[0]= "blank1"
            track = 0
            for x in lines[1]:
                if (x != ' '):
                    track += 1
            if (track == 0):
                print(lines)
                lines[1] = "blank2"
            if (length != 3):
                print(lines)
                lines.append(line[0].lower())
                lines.append("<pad>")
                lines.append(line[1])
            else:
                lines.append(line[0].lower())
                lines.append(line[1].lower())
                lines.append(line[2])

            res.append([lines[0].lower(), lines[1].lower(), float(lines[2])])
            #res.append([lines[3], lines[4], float(lines[0])])

    return res
x=read_data("trec_all_raw.txt")
#for i in x:
#    print(i)
d_model=768
str1=""
str2=""
j=0
for i in x:
    #r=["<pad>"]*(35)
    i[0] = i[0].split()
    i[1] = i[1].split()
    #i[0] = i[0]+r
    #i[1] = i[1]+r
    #i[0] = i[0][:30]
    #i[0] = i[1][:30]
    separator=' '
    temp1 = separator.join(i[0])
    temp2 = separator.join(i[1])
    str1+=temp1+"\n"
    str2+=temp2+"\n"

str1 = str1.split('\n')
str2 = str2.split('\n')
print(len(str1))

ctx = mx.gpu(0)
bert_embedding = BertEmbedding(ctx=ctx)
#bert_embedding = BertEmbedding(ctx=ctx, model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_uncased')

j=0
hf1=h5py.File('berttrecr1.h5','w')
ii=0
for i in range(0,56082,32):
    result = bert_embedding(str1[i:min(i+32,56082)])
    length = len(result)
    #print(length)
    for j in range(0,length):
        r=result[j][1:]
        r=np.array(r)
        val = 25 - int((r.size) / d_model)
        y = np.zeros((1, val, d_model))
        yy = r
        xx=np.append(yy, y, axis=1)
        if(j==0):
            x=xx
        else:
            x = np.append(x, xx, axis=0)
    hf1.create_dataset('dataset_' + str(i), data=x)
    f1.write(str(ii) + " " + str(i) + "\n")
    ii=ii+1
j=0
hf1.close()


ii=0
j=0

hf2 = h5py.File('berttrecr2.h5', 'w')
for i in range(0,56082,32):
    result = bert_embedding(str2[i:min(i+32,56082)])
    length = len(result)
    for j in range(0,length):
        r=result[j][1:]
        r=np.array(r)
        val = 25 - int((r.size) / d_model)
        y = np.zeros((1, val, d_model))
        yy = r
        xx = np.append(yy, y, axis=1)
        if(j==0):
            x=xx
        else:
            x = np.append(x, xx, axis=0)
    hf2.create_dataset('dataset_' + str(i), data=x)
    f2.write(str(ii)+" "+str(i)+"\n")
    ii=ii+1
hf2.close()

