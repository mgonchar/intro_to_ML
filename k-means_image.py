'''
Created on 22 июня 2016 г.

@author: miroslvgoncarenko
'''

from skimage.io import imread
from skimage import img_as_float
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np
import numpy.matlib
import pylab

def ind2sub(idx,width):
    return (idx//width, idx%width)

def PSNR_float_range(img1,img2):
    return -10*np.log10(mean_squared_error(img1.reshape(img1.shape[0]*img1.shape[1]),img2.reshape(img2.shape[0]*img2.shape[1])))

image = imread('parrots.jpg')
img = img_as_float(image)

shp = img.shape
img = img.reshape((shp[0]*shp[1],shp[2]))

for i in range(1,12):
    clst = KMeans(n_clusters=i, init='k-means++', random_state=241)
    labels = clst.fit_predict(img)
    
    img_mean = np.copy(img)
    for clst_i in range(0,i):
        idxs = np.where(labels==clst_i)[0]
        tmp = np.zeros(3)
        np.mean(img[idxs], axis=0, out=tmp)
        img_mean[idxs] = np.matlib.repmat(tmp,len(idxs),1)
        sht = True

    cur_PSNR = PSNR_float_range(img_mean,img)
    print("with "+str(i)+" clusters got PSNR: " + str(cur_PSNR)+"\n")
    
    shp_img = img_mean.reshape(shp)
    pylab.imshow(shp_img)
    pylab.show()
    shft = True

shuffle = True