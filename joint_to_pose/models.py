import tensorflow as tf
import keras



def basic(embedding,outputsize):

    embedding = keras.layers.Flatten()(embedding)
    #layer = keras.layers.Dense(64, activation=tf.nn.relu6)(embedding)
    layer = keras.layers.Dense(32, activation=tf.nn.relu6)(embedding)
    outputs = keras.layers.Dense(outputsize, activation="softmax")(layer)
    
    return outputs

def DNN(embedding,outputsize):
    
    embedding = keras.layers.Flatten()(embedding)
    layer = keras.layers.Dense(256, activation=tf.nn.relu)(embedding)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Dropout(0.4)(layer)

    layer = keras.layers.Dense(128, activation=tf.nn.relu)(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Dropout(0.4)(layer)

    layer = keras.layers.Dense(64, activation=tf.nn.relu)(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Dropout(0.4)(layer)

    outputs = keras.layers.Dense(outputsize, activation="softmax")(layer)
    
    return outputs

def CNN(embedding,outputsize):
    conv1 = keras.layers.Conv1D(64, kernel_size=3, activation=tf.nn.relu)(embedding)
    conv2 = keras.layers.Conv1D(64, kernel_size=3, activation=tf.nn.relu)(conv1)
    maxpool = keras.layers.MaxPooling1D(pool_size=2)(conv2)

    # Flatten the output of the CNN layers
    flatten = keras.layers.Flatten()(maxpool)

    # Dense layers
    dense1 = keras.layers.Dense(128, activation=tf.nn.relu)(flatten)
    dropout1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(64, activation=tf.nn.relu)(dropout1)
    dropout2 = keras.layers.Dropout(0.5)(dense2)

    # Output layer
    outputs = keras.layers.Dense(outputsize, activation="softmax")(dropout2)
    return outputs