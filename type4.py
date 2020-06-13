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

root = "D:\\projects\\BlenderAutoAnimator"
#TrainData = load_files("D:\\projects\\BlenderAutoAnimator\\Train")
TrainData = "D:\\projects\\BlenderAutoAnimator\\TrainData2\\"
filenames = []
#x = [[[[]for i in range(4)] for i in range(67)] ]
x = []
y = []
X1 = []
X2 = []
z = []


for target in os.listdir(TrainData):
    print(target)
    for f in os.listdir(os.path.join(TrainData+"/"+target)):
        o_data = (np.loadtxt(TrainData+"/"+target+"/"+f))
        o_data = o_data.reshape(int(o_data.shape[0]/67),67,4)
        initial = o_data.shape[0]
        if(initial<=300):
            data = o_data
            additionals = int((300-initial)%initial)
            for i in range(int((300-initial)/initial)):
                data = np.vstack((data,o_data))
            data = np.vstack((data,o_data[:additionals]))
            data = tf.image.resize(data,[48,48])
            X1.append(data)
            z.append(target)

BUFFER_SIZE = 6
BATCH_SIZE = 2
X1 = np.asarray(X1)
print(X1.shape)
X1 = X1.reshape(X1.shape[0], 48, 48, 4, 1).astype('float32')
max_ = tf.math.reduce_max(X1)
min_ = tf.math.reduce_min(X1)
X1 = (X1 - min_) / (max_-min_) # Normalize the images to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(X1).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

"""
#for element in train_dataset:
#    print(element.shape[0])
"""

# Generator Code
def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(12*12*2*100, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((12,12,2,100)))
    assert model.output_shape == (None, 12,12,2,100)

    #Required (10,16,2) same:padding
    model.add(layers.Conv3DTranspose(512, (3, 3, 3), strides=(1, 1, 2), padding="same", use_bias=False))
    assert model.output_shape == (None, 12,12,4,512)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    #Required (20,32,4) same:padding
    model.add(layers.Conv3DTranspose(256, (2, 2, 2), strides=(1, 1, 1), padding="same", use_bias=False))
    assert model.output_shape == (None, 12,12,4,256)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    #Required (40,32,4) same:padding
    model.add(layers.Conv3DTranspose(128, (2, 2, 2), strides=(1, 1, 1), padding="same", use_bias=False))
    assert model.output_shape == (None, 12,12,4,128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    #Required (20,32,4) same:padding
    model.add(layers.Conv3DTranspose(64, (2, 2, 2), strides=(1, 1, 1), padding="same", use_bias=False))
    assert model.output_shape == (None, 12,12,4,64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    #Required (99,32,4) valid:padding    (3,2,1)
    model.add(layers.Conv3DTranspose(1, (6, 6, 6), strides=(4, 4, 1), padding="same", use_bias=False, activation="tanh"))
    assert model.output_shape == (None, 48,48,4,1) #final must be (300,67,4,1)
    return model

generator = generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# Discriminator Code
def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(64, (2, 2, 2), strides=(1, 1, 1), padding="same", input_shape=[48, 48, 4, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(128, (2, 2, 2), strides=(1, 1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(256, (2, 2, 2), strides=(1, 1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(512, (4, 4, 4), strides=(2, 2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    return model

discriminator = discriminator_model()
decision = discriminator(generated_image)

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

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    predictions = (predictions[0, :, :, :, 0] * (max_-min_)) + min_
    predictions = tf.image.resize(predictions,[300,67])
    predictions = np.asarray(predictions).reshape(300,67,4)
    print(predictions)
    #fig = plt.figure(figsize=(4,4))

    with open("D:\\projects\\BlenderAutoAnimator\\Test\\test23.txt", 'w') as outfile:
        outfile.write('#Array shape: {0}\n'.format(predictions.shape))
        for data_slice in predictions:
            np.savetxt(outfile,data_slice)
            outfile.write('#New slice\n')

train(train_dataset, EPOCHS)