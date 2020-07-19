from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
import h5py

def read_data(filename):
    with open(filename, 'r') as datafile:
        res = []
        for line in datafile:
            line = line.strip().split('\t')
            lines = []
            length = len(line)
            if (length < 3):
                lines.append(line[0])
                lines.append("<pad>")
                lines.append(line[1])
            else:
                lines.append(line[0])
                lines.append(line[1])
                lines.append(line[2])

            res.append([lines[0], lines[1], float(lines[2])])

    return res
x=read_data("semeval_2015_all.txt")
temp1=[]
temp2=[]
j=0
for i in x:
    r=["<pad>"]*(25)
    i[0] = i[0].split()
    i[1] = i[1].split()
    i[0] = i[0]+r
    i[1] = i[1]+r
    temp1.append(i[0][:25])
    temp2.append(i[1][:25])
#print((temp1))

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


j=0
k=j
elmo10=[]
elmo11=[]

hf1=h5py.File('elmosemeval20151.h5','w')

for i in range(0,20162):
    k=j+32
    k=min(k,20162)
    elmo = Elmo(options_file, weight_file, 2, dropout=0)
    sentences = temp1[j:k]
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)
    try:
        tempelmo10 = embeddings['elmo_representations'][0].detach().numpy()
        #tempelmo21 =embeddings['elmo_representations'][1].detach().numpy()
    except:
         k=j-32
         k = min(k, 20162)
         continue
    #tempelmo11 = embeddings['elmo_representations'][1].detach().numpy()
    hf1.create_dataset('dataset_'+str(j),data=tempelmo10)
    '''
    if (j == 0):
        elmo10 = tempelmo10
        elmo11 = tempelmo11
    else:
        elmo10 = np.append(elmo10, tempelmo10, axis=0)
        elmo11 = np.append(elmo11, tempelmo11, axis=0)
    '''
    #if (j % 1000 == 0):
    #    print(str(j)+":1")
    j = j + 32
    if(j>=20162):
        break
hf1.close()
print("hf1 done")

j=0
k=j
elmo20=[]
elmo21=[]
temp1=""

hf2=h5py.File('elmosemeval20152.h5','w')
for i in range(0,20162):
    k=j+32
    k=min(k,20162)
    elmo = Elmo(options_file, weight_file, 2, dropout=0)
    sentences = temp2[j:k]
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)
    try:
         tempelmo20 = embeddings['elmo_representations'][0].detach().numpy()
         #tempelmo21 =embeddings['elmo_representations'][1].detach().numpy()
    except:
         k=j-32
         k = min(k,20162)
         continue
    hf2.create_dataset('dataset_' + str(j), data=tempelmo20)

    '''
    if (j == 0):
        elmo20 = tempelmo20
        elmo21 = tempelmo21
    else:
        elmo20 = np.append(elmo20, tempelmo20, axis=0)
        elmo21 = np.append(elmo21, tempelmo21, axis=0)
    '''
    #if(j%1000==0):
        #print(str(j)+":2")
    j = j + 32
    if(j>=20162):
          break
hf2.close()
#print(embeddings['elmo_representations'][0])
'''
outfile10 = "Quoraelmo1(10)"
outfile11 = "Quoraelmo2(11)"
outfile20 = "Quoraelmo1(20)"
outfile21 = "Quoraelmo2(21)"
np.save(outfile10, elmo10)
np.save(outfile11, elmo21)
np.save(outfile20, elmo20)
np.save(outfile21, elmo21)
'''
#print(elmo10.shape)
#print(elmo11.shape)
#print(elmo20.shape)
#print(elmo21.shape)

#print(j)
'''
from allennlp.commands.elmo import ElmoEmbedder
elmo = ElmoEmbedder()
tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
vectors = elmo.embed_sentence(tokens)
assert(len(vectors) == 3) # one for each layer in the ELMo output
assert(len(vectors[0]) == len(tokens)) # the vector elements correspond with the input tokens
import scipy
vectors2 = elmo.embed_sentence(["I", "ate", "a", "carrot", "for", "breakfast"])
#scipy.spatial.distance.cosine(vectors[2][3], vectors2[2][3]) # cosine distance between "apple" and "carrot" in the last layer
#0.18020617961883545
print(vectors2.size)
'''

print("hf2 done")