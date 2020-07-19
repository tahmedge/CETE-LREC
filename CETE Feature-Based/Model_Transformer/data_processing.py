
f1=open("../data/wikitrainnewline.txt", 'r')
f2=open("../data/wikitestnewline.txt", 'r')
f3=open("../data/wikidevnewline.txt", 'r')
res=""
for line in f1:
    res+=(str(line))
for line in f2:
    res += (str(line))
for line in f3:
    res += (str(line))
x=0
f4 = open("../wiki_not_aligned_data.txt", 'w')
f4.write(res)
#for i in res:
#print(x)