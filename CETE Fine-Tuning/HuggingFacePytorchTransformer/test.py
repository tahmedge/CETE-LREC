type="train"
f=open("semeval2015_"+type+".tsv","r")
r=0
temp=set()
for i in f:
    k=0
    r=r+1
    '''
    for j in i:
        if(j=='\t'):
            k=k+1
    if(k!=2):
        print(r)
    '''
    p=i.split('\t')
    temp.add(p[0])
print(len(temp))
