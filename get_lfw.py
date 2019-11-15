import keras
import glob
import os
from scipy.misc import imread
import numpy as np
import h5py

tgz_path = keras.utils.get_file('lfw-deepfunneled.tgz','http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz',extract=True)
tgz_file_dir, tgz_file_path = os.path.split(tgz_path)
dir_path = tgz_file_dir + '/' + tgz_file_path.split('.')[0]

dirs = sorted(glob.glob(dir_path + '/*'))

num_dirs = len(dirs)
num_imgs = 0
for dir_name in dirs:
    num_imgs += len(glob.glob(dir_name + '/*.jpg'))

x = np.zeros((num_imgs,128,128,3),dtype='uint8')
y = np.zeros((num_imgs,),dtype='int32')

d = 0
n = 0
for dir_name in dirs:
    for img_name in sorted(glob.glob(dir_name + '/*.jpg')):
        img = imread(img_name)
        img = img[61:189,61:189,:]
        x[n] = img
        #y[n] = d
        # set class to letter of alphabet of first name
        first_char = os.path.basename(img_name)[0]
        y[n] = ord(first_char) - ord('A')
        n += 1
    d += 1

f = h5py.File('lfw.hdf5','w')
f.create_dataset('data',data=x)
f.create_dataset('label',data=y)
f.close()

