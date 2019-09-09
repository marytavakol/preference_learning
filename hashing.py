# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:48:57 2017

@author: maryam
"""

import numpy as np
import ujson
import sys
from sklearn.feature_extraction import FeatureHasher
from sklearn.datasets import dump_svmlight_file

#@profile
def convert_data():
    items = []
    with open(str(sys.argv[2]), 'r') as fd:
        lines = fd.readlines()

    for line in lines:
        items.append(ujson.loads(line))
        
    hash_type = str(sys.argv[4])
    print('num of items: ', len(items))
    sys.stdout.flush()
    Xp = []
    Xn = []
    h = FeatureHasher(n_features=nf, input_type='string', alternate_sign=True) # The default is True, but I put it again to make sure we use the right version

    with open(str(sys.argv[1]), 'r') as fd:
        lines = fd.readlines()

    for line in lines:
        line = ujson.loads(line)
        usr = line['user']
        fc = items[int(line['context'])]
        win = items[int(line['winner'])]
        los = items[int(line['loser'])]
        xp = []
        xn = []
        if hash_type == 'avg':
            for k1 in fc:
                for k2 in win:
                    xp.append(fc[k1] + win[k2])
                    xn.append(fc[k1] + los[k2])
        elif hash_type == 'prs':
            for k1 in fc:
                for k2 in win:
                    xp.append(usr + fc[k1] + win[k2])
                    xn.append(usr + fc[k1] + los[k2])
        elif hash_type == 'bth':
            for k1 in fc:
                for k2 in win:
                    xp.append(usr + fc[k1] + win[k2])
                    xp.append(fc[k1] + win[k2])
                    xn.append(usr + fc[k1] + los[k2])
                    xn.append(fc[k1] + los[k2])
        else:
            print('wrong hash type!\n')
            sys.exit(0)

        Xp.append(xp)
        Xn.append(xn)

    del items
    n = len(Xp)
    d = len(Xp[0])
    print('num of data points: ', n)
    print('dim of features: ', d)
    sys.stdout.flush()

    f1 = h.transform(Xp)#.toarray()
    f2 = h.transform(Xn)#.toarray()
    X = f1 - f2

    y = [-1, 1]*(n/2)
    if n%2 == 1:
        y.append(-1)
    yy = np.reshape(y, (n, 1))
    X = X.multiply(yy)
    print('writing to file...')
    dump_svmlight_file(X, y, sys.argv[3], zero_based=False)

   
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print ('enter the input (data+features) and output file names + type:1:avg,2:prs,3:both.')
        sys.exit(0)

    nf = 100000#int(sys.argv[4])

    convert_data()
   
    print('done')



