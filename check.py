# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 13:14:22 2019

@author: Dell
"""

from mnist import MNIST
mndata = MNIST('samples')

images, labels = mndata.load_training()
# or
images, labels = mndata.load_testing()