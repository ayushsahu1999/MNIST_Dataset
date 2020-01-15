# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 23:25:06 2020

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from image_data import images

ims = images('samples/train-images-idx3-ubyte')

minimum = np.amin(ims[0])
maximum = np.amax(ims[0])
PI = 3.14


def normalization(inp, mins, maxs):
    x_norm = (inp - mins) / (maxs - mins)
    return x_norm

def gaussian(sig, u, x):
    x_gaus = (1/(sig*((2*PI)**(1/2))))*np.exp(-((x-u)**2)/(2*(sig**2)))
    return x_gaus

def scaling(a, minAllowed, maxAllowed, _min, _max):
    val = (maxAllowed-minAllowed) * (a - _min) / (_max - _min) + minAllowed
    return val
    
img = ims[0].copy()
#print (ims[0].size)

#for i in range(len(ims[0])):
#    for j in range(len(ims[0][i])):
        #img[i][j] = normalization(ims[0][i][j], minimum, maximum)
        #img[i][j] = gaussian(std, mean, ims[0][i][j])
        
print ('Before: '+str(np.amin(img)))
print ('Before: '+str(np.amax(img)))
print (np.mean(img))
print (np.std(img))

#sns.distplot(img.ravel(), hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'dist')

img = cv2.GaussianBlur(img, (5,5), 0)
img = cv2.adaptiveThreshold(img,                          
                            255,                                  
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       
                            cv2.THRESH_BINARY_INV,                
                            11,
                            2)

plt.figure(figsize=(28, 28))
plt.imshow(np.array(img))
plt.show()

# Now distributing the data with mean=0 and standard deviation=1
# z = (x - u) / s
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
img = scaler.fit_transform(img)
        

#sns.distplot(img.ravel(), hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'dist')
#plt.plot(img.ravel(), norm.pdf(img.ravel(), 0, 1), color='blue')
#plt.title('Gaussian Distribution')
#plt.show()
hist, bin_edges = np.histogram(img.ravel())
_ = plt.hist(img.ravel(), bins='auto')
plt.show()

print ('After: '+str(np.amin(img)))
print ('After: '+str(np.amax(img)))
print (np.mean(img))
print (np.std(img))