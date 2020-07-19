# coding: utf-8
def calc_one_map(data):
    relcnt = 0
    score = 0.0
    data = sorted(data, key=lambda d: d[1], reverse=True)
    for idx, item in enumerate(data):
        if float(item[0][2]) == 1:
            relcnt = relcnt + 1
            score = score + 1.0 * relcnt / (idx + 1)
    if relcnt == 0:
        return 0
    return score / relcnt


def calc_one_mrr(data):
    score = 0
    data = sorted(data, key=lambda d: d[1], reverse=True)
    for idx, item in enumerate(data):
        if float(item[0][2]) == 1:
            score = 1.0 / (idx + 1)
            break
    return score


def calc_one_precision(data):
    score = 0
    data = sorted(data, key=lambda d: d[1], reverse=True)
    item = data[0]
    if float(item[0][2]) == 1:
        score = 1.0
    return score / 1.0


def calc_one(data):
    relcnt = 0
    score = 0.0
    data = sorted(data, key=lambda d: d[1], reverse=True)
    fout=open("meshres.5.txt", 'a')
    for idx, item in enumerate(data):
        if idx < 5:
            fout.write(item[0][0] + '\t' + item[0][1] + '\n')


def calc_map(testfile, predfile, resfile):
    with open(testfile, 'r') as ftest, open(predfile, 'r') as fpred, open(resfile, 'w') as fout:
        data = []
        pred = []
        n = 0
        for line in ftest:
            data.append(line.strip().split('\t'))
        for line in fpred:
            try:
                pred.append(float(line.strip()))  
                n += 1
            except Exception:
                #print n
                #print line
                print("hello")
        oneq = []
        pre = "BEGIN"
        mapscore = 0.0
        excnt = 0
        for item in zip(data, pred):
            if item[0][0] == pre or pre == "BEGIN":
                oneq.append(item)
            else :
                excnt = excnt + 1
                sc = calc_one_map(oneq)
                fout.write(str(sc) + '\n')
                if sc == 0:
                    excnt = excnt - 1
                mapscore = mapscore + sc
                oneq = []
                oneq.append(item)
            pre = item[0][0]
        sc = calc_one_map(oneq)
        fout.write(str(sc) + '\n')
        if sc != 0:
            excnt = excnt + 1
        mapscore = mapscore + sc
        fout.write(str(mapscore / excnt) + '\n')
        return mapscore / excnt


def calc_mrr(testfile, predfile, resfile):
    with open(testfile, 'r') as ftest, open(predfile, 'r') as fpred, open(resfile, 'w') as fout:
        data = []
        pred = []
        n = 0
        for line in ftest:
            data.append(line.strip().split('\t'))
        for line in fpred:
            try:
                pred.append(float(line.strip()))  
                n += 1
            except Exception as e:
                #print n
                #print line
                print(e)
                
        oneq = []
        pre = "BEGIN"
        mrrscore = 0.0
        excnt = 0
        for item in zip(data, pred):
            if item[0][0] == pre or pre == "BEGIN":
                oneq.append(item)
            else :
                excnt = excnt + 1
                sc = calc_one_mrr(oneq)
                fout.write(str(sc) + '\n')
#                 print sc,
                if sc == 0:
                    excnt = excnt - 1
                mrrscore = mrrscore + sc
                oneq = []
                oneq.append(item)
            pre = item[0][0]
        sc = calc_one_mrr(oneq)
        fout.write(str(sc) + '\n')
#         print sc,
        if sc != 0:
            excnt = excnt + 1
        mrrscore = mrrscore + sc
        fout.write(str(mrrscore / excnt) + '\n')
        return mrrscore / excnt


def calc_map1(testfile, preds):
    with open(testfile, 'r') as ftest:
        data = []
        pred = []
        for line in ftest:
            data.append(line.strip().split('\t'))
        for p in preds:
            pred.append(float(p))
        oneq = []
        pre = "BEGIN"
        mapscore = 0.0
        excnt = 0
        for item in zip(data, pred):
            if item[0][0] == pre or pre == "BEGIN":
                oneq.append(item)
            else :
                excnt = excnt + 1
                sc = calc_one_map(oneq)
                if sc == 0:
                    excnt = excnt - 1
                mapscore = mapscore + sc
                oneq = []
                oneq.append(item)
            pre = item[0][0]
        sc = calc_one_map(oneq)
        if sc != 0:
            excnt = excnt + 1
        mapscore = mapscore + sc
        return mapscore / excnt



def calc_mrr1(testfile, preds):
    with open(testfile, 'r') as ftest:
        data = []
        pred = []
        n = 0
        for line in ftest:
            data.append(line.strip().split('\t'))         
        for p in preds:
            pred.append(float(p))
                
        oneq = []
        pre = "BEGIN"
        mrrscore = 0.0
        excnt = 0
        for item in zip(data, pred):
            if item[0][0] == pre or pre == "BEGIN":
                oneq.append(item)
            else :
                excnt = excnt + 1
                sc = calc_one_mrr(oneq)
#                 print sc,
                if sc == 0:
                    excnt = excnt - 1
                mrrscore = mrrscore + sc
                oneq = []
                oneq.append(item)
            pre = item[0][0]
        sc = calc_one_mrr(oneq)
#         print sc,
        if sc != 0:
            excnt = excnt + 1
        mrrscore = mrrscore + sc
        return mrrscore / excnt
    

def calc_precision1(testfile, preds):
    with open(testfile, 'r') as ftest:
        data = []
        pred = []
        n = 0
        for line in ftest:
            data.append(line.strip().split('\t'))         
        for p in preds:
            pred.append(float(p))
            
        oneq = []
        pre = "BEGIN"
        pscore = 0.0
        excnt = 0
        for item in zip(data, pred):
            if item[0][0] == pre or pre == "BEGIN":
                oneq.append(item)
            else :
                excnt = excnt + 1
                sc = calc_one_precision(oneq)
                pscore = pscore + sc
                oneq = []
                oneq.append(item)
            pre = item[0][0]
        sc = calc_one_precision(oneq)
        excnt = excnt + 1
        pscore = pscore + sc
        return pscore / excnt    

def calc_map_mesh(testfile, predfile):
    with open(testfile, 'r') as ftest, open(predfile, 'r') as fpred:
        data = []
        pred = []
        for line in ftest:
            data.append(line.strip().split('\t'))
        for line in fpred:
            pred.append(float(line.strip()))
        oneq = []
        pre = "BEGIN"
        mapscore = 0.0
        excnt = 0
        resu=zip(data, pred)
        resu = sorted(resu, key=lambda d: d[1], reverse=True)
        fout = open('mfwtest.out7.txt', 'w')
        for item in resu:
            fout.write(item[0][0] + '\t' + item[0][1] + '\t' + str(item[1]) + '\n')







