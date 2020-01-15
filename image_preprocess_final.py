# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 23:25:44 2020

@author: Dell
"""

def img_pre(path):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from image_data import images
    from sklearn.preprocessing import StandardScaler
    
    ims = images(path)
    #ims = images('samples/train-images-idx3-ubyte')
    
    minimum = np.amin(ims[0])
    maximum = np.amax(ims[0])
    PI = 3.14
    
        
    imgs = ims.copy()
    #print (ims[0].size)
    print (imgs[1].shape)
    scaler = StandardScaler()
    
    dataset = []
    
    for i in range(len(imgs)):
        img = imgs[i].copy()
        #mean = np.mean(ims[i])
        #std = np.std(ims[i])
        #print (mean)
        #print (std)
        
        #for i in range(len(ims[0])):
        #    for j in range(len(ims[0][i])):
                #img[i][j] = normalization(ims[0][i][j], minimum, maximum)
                #img[i][j] = gaussian(std, mean, ims[0][i][j])
                
        #print ('Before: '+str(np.amin(img)))
        #print ('Before: '+str(np.amax(img)))
        #print (np.mean(img))
        #print (np.std(img))
        
        #sns.distplot(img.ravel(), hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'dist')
        
        img = cv2.GaussianBlur(img, (5,5), 0)
        img = cv2.adaptiveThreshold(img,                          
                                    255,                                  
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       
                                    cv2.THRESH_BINARY_INV,                
                                    11,
                                    2)
        
        #plt.figure(figsize=(28, 28))
        #plt.imshow(np.array(img))
        #plt.show()
        
        # Now distributing the data with mean=0 and standard deviation=1
        # z = (x - u) / s
        
        scaler.fit(img)
        img = scaler.transform(img)
        
        dataset.append(img)
                
        if (i < 10):
            print (np.mean(np.array(dataset[i])), np.std(np.array(dataset[i])))
        #sns.distplot(img.ravel(), hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'dist')
        #plt.plot(img.ravel(), norm.pdf(img.ravel(), 0, 1), color='blue')
        #plt.title('Gaussian Distribution')
        #plt.show()
        #hist, bin_edges = np.histogram(img.ravel())
        #_ = plt.hist(img.ravel(), bins='auto')
        #plt.show()
        
        #print ('After: '+str(np.amin(img)))
        #print ('After: '+str(np.amax(img)))
        #print (np.mean(img))
        #print (np.std(img))
            
    dataset = np.array(dataset)
    return dataset

#cds = img_pre('samples/train-images-idx3-ubyte')