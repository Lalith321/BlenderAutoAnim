import tensorflow as tf
from tensorflow.keras import layers
import numpy as np  
import os,sys,time
import sklearn as sk
from IPython import display
import matplotlib.pyplot as plt
#from sklearn.preprocessing import train_test_split
from sklearn.datasets import load_files
from tensorflow_core._api.v2.compat.v1.random.experimental import Generator
from numba import njit,jit,vectorize

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
"""
root = "D:\\projects\\BlenderAutoAnimator"
#TrainData = load_files("D:\\projects\\BlenderAutoAnimator\\Train")
TrainData = "D:\\projects\\BlenderAutoAnimator\\TrainData2"
filenames = []
#x = [[[[]for i in range(4)] for i in range(67)] ]
x = []
y = []
X1 = np.array([],dtype=np.int64).reshape(0,48,48,48)
X2 = []
z = []

def preprocess1(data):
    global X1
    data = tf.reshape(data,[300,268,1])
    data = tf.image.resize(data,[48,2304])
    data = tf.reshape(data,[1,48,48,48]) # Reshaping the data
    #X2.append(data)
    X1 = np.append(X1,data,axis = 0)
    return X1

for target in os.listdir(TrainData):
    print(target)
    for f in os.listdir(os.path.join(TrainData+"\\"+target)):
        o_data = (np.loadtxt(TrainData+"\\"+target+"\\"+f))
        o_data = o_data.reshape(int(o_data.shape[0]/67),67,4) # Turn the image to (300, 67, 4)
        initial = o_data.shape[0]
        if(initial<=300):
            data = o_data
            additionals = int((300-initial)%initial) # If image.shape[1] is less than 300 then add the extra data starting from frame 1 (10's digit)
            for i in range(int((300-initial)/initial)):
                data = np.vstack((data,o_data)) # additional daata (100's digit)
            data = np.vstack((data,o_data[:additionals])) # Create a bunch of data by stacking it one on the other
            # Special steps for resizing 3D data
            preprocess1(data)"""

root = "D:\\projects\\BlenderAutoAnimator"

