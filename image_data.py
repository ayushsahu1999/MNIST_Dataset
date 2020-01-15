# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:44:18 2020

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import gzip

# Getting the images
def images(path):
    # path = 'samples/train-images-idx3-ubyte'
    file = path
    arr = idx2numpy.convert_from_file(file)
    return arr

#plt.figure(figsize=(28, 28))
#plt.imshow(arr[3])
#plt.show()

#print (type(arr[1]))

# Getting the labels
#f = gzip.open('compressed_mnist/train-labels-idx1-ubyte.gz','r')
#f.read(8)
#for i in range(0,50):   
#    buf = f.read(1)
#    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
#    print(labels)