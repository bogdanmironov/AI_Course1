import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.6):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# np.set_printoptions(linewidth=200)

# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])

training_images  = training_images / 255.0
test_images = test_images / 255.0

callbacks = myCallback()

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), 
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10)

model.evaluate(test_images, test_labels)