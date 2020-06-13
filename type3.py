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

"""device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))"""


root = "D:\\projects\\BlenderAutoAnimator"
#TrainData = load_files("D:\\projects\\BlenderAutoAnimator\\Train")
TrainData = "D:\\projects\\BlenderAutoAnimator\\Train"
filenames = []
#x = [[[[]for i in range(4)] for i in range(67)] ]
x = []
y = []
X1 = []
X2 = []
z = []

for target in os.listdir(TrainData):
    print(target)
    for f in os.listdir(os.path.join(TrainData+"\\"+target)):
        if(target=="combined"):
            data = (np.loadtxt(TrainData+"\\"+target+"\\"+f)).reshape(300,67,4)
            X1.append(data)
            z.append(target)
        else:
            data = (np.loadtxt(TrainData+"\\"+target+"\\"+f)).reshape(325,67,4)
            X2.append(data)
            #z.append(target)
        xas=np.asarray(data).shape
        for i in range (xas[0]):
            x.append(data[i])
            y.append(target)

BUFFER_SIZE = 6
BATCH_SIZE = 2
X1 = np.asarray(X1)
X1 = X1.reshape(X1.shape[0], 300, 67, 4, 1).astype('float32')
max_ = tf.math.reduce_max(X1)
min_ = tf.math.reduce_min(X1)
X1 = (X1 - min_) / (max_-min_) # Normalize the images to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(X1).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

"""for element in train_dataset:
    print(element.shape[0])
"""

# Generator Code
def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(33*16*2*100, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((33,16,2,100)))
    assert model.output_shape == (None, 33,16,2,100)

    #Required (10,16,2) same:padding
    model.add(layers.Conv3DTranspose(128, (2, 4, 3), strides=(3, 2, 2), padding="same", use_bias=False))
    assert model.output_shape == (None, 99,32,4,128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    #Required (20,32,4) same:padding
    model.add(layers.Conv3DTranspose(64, (2, 2, 2), strides=(1, 1, 1), padding="same", use_bias=False))
    assert model.output_shape == (None, 99,32,4,64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    #Required (99,32,4) valid:padding    (3,2,1)
    model.add(layers.Conv3DTranspose(1, (6, 5, 1), strides=(3, 2, 1), padding="valid", use_bias=False, activation="tanh"))
    assert model.output_shape == (None, 300,67,4,1) #final must be (300,67,4,1)
    return model

generator = generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# Discriminator Code
def discriminator_model():
    visible1 = layers.Input(shape=(300,57,4,1))
    conv11 = layers.Conv3D(64, kernel_size=5, activation='leakyrelu', dropout=0.3)(visible1)
    conv12 = layers.Conv3D(128, kernel_size=5, activation='leakyrelu', dropout=0.3)(conv11)
    conv13 = layers.Conv3D(256, kernel_size=5, activation='leakyrelu', dropout=0.3)(conv12)
    flat1 = layers.Flatten()(conv13)

    visible2 = layers.Input(shape=(300,10,4,1))
    conv21 = layers.Conv3D(32, kernel_size=2, activation='relu')(visible2)
    flat2 = layers.Flatten()(conv21)

    merge = layers.concatenate([flat1,flat2])
    hidden1 = layers.Dense(10, activation='relu')(merge)
    output = layers.Dense(1,activation='relu')(hidden1)
    model = tf.keras.models.Model(inputs=[visible1, visible2], outputs=output)
    return model

discriminator = discriminator_model()
decision = discriminator([generated_image[:,:,:57,:],generated_image[:,:,57:,:]])

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,beta_1=0.5)#,beta_2=0.999,epsilon=1e-07)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,beta_1=0.5)#,beta_2=0.999,epsilon=1e-07)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

EPOCHS = 500
noise_dim = 100
num_examples_to_generate = 1
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator([images[:,:,:57,:],images[:,:,57:,:]], training=True)
        fake_output = discriminator([generated_images[:,:,:57,:],generated_images[:,:,57:,:]], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs, manager, checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    if tf.train.latest_checkpoint(checkpoint_dir):
        print("Restored from {}" .format(manager.latest_checkpoint))
    else:
        print("Initializing from Scratch")

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print ('Optimizer :')
    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    predictions = (predictions[0, :, :, :, 0] * (max_-min_)) + min_
    predictions = np.asarray(predictions).reshape(300,67,4)
    print(predictions)
    #fig = plt.figure(figsize=(4,4))

    with open("D:\\projects\\BlenderAutoAnimator\\Test\\private1.txt", 'w') as outfile:
        outfile.write('#Array shape: {0}\n'.format (predictions.shape))
        for data_slice in predictions:
            np.savetxt(outfile,data_slice)
            outfile.write('#New slice\n')

train(train_dataset, EPOCHS, manager, checkpoint_dir)