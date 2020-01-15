# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:32:11 2020

@author: Dell
"""
import gzip
import numpy as np

def train_labels(path):
    # Getting the labels
    f = gzip.open(path,'r')
    f.read(8)
    labels = []
    
    try:
        for i in range(60000):
            buf = f.read(1)
            l = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            if (l is None):
                break
            labels.append(l)
            
        labels = np.array(labels)
        
    except EOFError:
        pass
        
    print (labels.shape)
    labels = np.squeeze(labels)
    return np.array(labels, dtype='float64')
    
def test_labels(path):
    # Getting the labels
    f = gzip.open(path,'r')
    f.read(8)
    labels = []
    
    try:
        for i in range(10000):
            buf = f.read(1)
            l = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            if (l is None):
                break
            labels.append(l)
            
        labels = np.array(labels)
        
    except EOFError:
        pass
        
    print (labels.shape)
    labels = np.squeeze(labels)
    return np.array(labels, dtype='float64')

#abc = test_labels('compressed_mnist/t10k-labels-idx1-ubyte.gz')