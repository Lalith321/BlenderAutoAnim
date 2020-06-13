#import tensorflow.compat.v1 as tf
"""
x = tf.random.uniform(shape = [2,3,4], minval=-1, maxval=1)
print(x,x.shape)
#tf.keras.utils.normalize(x, axis=1, order = 2)
#print(x,x.shape)
y = tf.math.reduce_max(x)
y1 = tf.math.reduce_min(x)
print(y,y1)
z = (x-y1)/(y-y1)
print(z)
z1 = (z*(y-y1))+y1
print(z1)

import os
print(os.path.splitext("Sample.jpg")[1])

import numpy as np

tf.disable_eager_execution()
a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")
b = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_b")
normalize_a = tf.nn.l2_normalize(a,0)        
normalize_b = tf.nn.l2_normalize(b,0)
cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
sess=tf.Session()
cos_sim=sess.run(cos_similarity,feed_dict={a:[37,47],b:[30,23]})
print(cos_sim)

import multiprocessing as mp
print(mp.cpu_count())
"""

# Multiple Inputs
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
import tensorflow as tf
# first input model
""" visible1 = Input(shape=(64,64,1))
conv11 = Conv2D(32, kernel_size=4, activation='relu')(visible1)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
conv12 = Conv2D(16, kernel_size=4, activation='relu')(pool11)
pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
flat1 = Flatten()(pool12)
# second input model
visible2 = Input(shape=(32,32,3))
conv21 = Conv2D(32, kernel_size=2, activation='relu')(visible2)
pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
conv22 = Conv2D(16, kernel_size=4, activation='relu')(pool21)
pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
flat2 = Flatten()(pool22)
# merge input models
merge = concatenate([flat1, flat2])
# interpretation model
hidden1 = Dense(10, activation='relu')(merge)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=[visible1, visible2], outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='multiple_inputs.png')

"""
class seperator(Layer):
    def __init__(self,ratio):
        super(seperator,self).__init__()
        self.ratio = ratio
    def call(self,inputs):
        #l = inputs[:,:,:int(inputs.shape[2]/ratio)+1,:]
        #m = inputs[:,:,inputs.shape[2]-int(inputs.shape[2]/ratio)-1:,:]
        l,_ = tf.split(inputs,[int(inputs.shape[2]/ratio)+1,inputs.shape[2]-int(inputs.shape[2]/ratio)-1],2)
        _,m = tf.split(inputs,[int(inputs.shape[2]/ratio)-1,inputs.shape[2]-int(inputs.shape[2]/ratio)+1],2)
        return l,m

ratio=67/57
x_seperator = seperator(ratio)

visible = layers.Input(shape=(48,48,48,1))

split01, split02 = x_seperator(visible)
conv11 = layers.Conv3D(32, kernel_size=5)(split01)
conv11 = layers.LeakyReLU()(conv11)
conv21 = layers.Conv3D(32, kernel_size=3)(split02)
conv21 = layers.LeakyReLU()(conv21)
conv22 = layers.Conv3D(32, kernel_size=3)(conv21)
conv22 = layers.LeakyReLU()(conv22)

merge1 = layers.concatenate([conv11,conv22],axis=2)

split11, split12 = x_seperator(merge1)
#split11, split12 = tf.split(merge1,[int(merge1.shape[2]/ratio),merge1.shape[2]-int(merge1.shape[2]/ratio)],2)
conv12 = layers.Conv3D(32, kernel_size=5)(split11)
conv12 = layers.LeakyReLU()(conv12)
conv23 = layers.Conv3D(32, kernel_size=3)(split12)
conv23 = layers.LeakyReLU()(conv23)
conv24 = layers.Conv3D(32, kernel_size=3)(conv23)
conv24 = layers.LeakyReLU()(conv24)

merge2 = layers.concatenate([conv12,conv24],axis=2)

split21, split22 = x_seperator(merge2)    
#split21, split22 = tf.split(merge2,[int(merge2.shape[2]/ratio),merge2.shape[2]-int(merge2.shape[2]/ratio)],2)
conv13 = layers.Conv3D(32, kernel_size=5)(split21)
conv13 = layers.LeakyReLU()(conv13)
conv25 = layers.Conv3D(32, kernel_size=3)(split22)
conv25 = layers.LeakyReLU()(conv25)
conv26 = layers.Conv3D(32, kernel_size=3)(conv25)
conv26 = layers.LeakyReLU()(conv26)

output = layers.concatenate([conv13,conv26],axis=2)

model = Model(inputs=[visible], outputs=output)
#print(model.summary())
plot_model(model,to_file="mine.png",show_shapes=True)
"""
from numba import njit
import numpy as np

def original_function(input_list):
    output_list = []
    for item in input_list:
        if item%2==0:
            output_list.append(2)
        else:
            output_list.append(1)
    return output_list

test_array = np.arange(10000)"""