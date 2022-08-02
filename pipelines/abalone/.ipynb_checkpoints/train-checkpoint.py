import argparse
import numpy as np
import os
import tensorflow as tf
import subprocess
import sys

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Softmax
from tensorflow.keras import optimizers


print(tf. __version__) 
print(np. __version__) 
print("************************************************************")
def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    # model directory
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()


def get_train_data(train_dir):

    x_train = np.load(os.path.join(train_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    print('x train', x_train.shape,'y train', y_train.shape)
    print(type(x_train))
    print(type(y_train))

    return x_train, y_train


def get_test_data(test_dir):

    x_test = np.load(os.path.join(test_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(test_dir, 'y_test.npy'))
    print('x test', x_test.shape,'y test', y_test.shape)
    print(type(x_test))
    print(type(y_test))

    return x_test, y_test



# def get_model():

#     inputs = tf.keras.Input(shape=(8,))
#     hidden_1 = tf.keras.layers.Dense(8, activation='relu')(inputs)
#     hidden_2 = tf.keras.layers.Dense(4, activation='relu')(hidden_1)
#     outputs = tf.keras.layers.Dense(1)(hidden_2)
#     return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_test_data(test_dir):
    model = Sequential()

    #Input Layer
    model.add(Dense(X.shape[1], activation='relu', input_dim = X.shape[1]))

    #Hidden Layer
    model.add(Dense(512,kernel_initializer='normal', activation='relu'))
    model.add(Dense(512,kernel_initializer='normal', activation='relu'))
    model.add(Dense(256,kernel_initializer='normal', activation='relu'))
    model.add(Dense(128,kernel_initializer='normal', activation='relu'))
    model.add(Dense(64,kernel_initializer='normal', activation='relu'))
    model.add(Dense(32,kernel_initializer='normal', activation='relu'))
    #Output Layer
    model.add(Dense(1,kernel_initializer='normal', activation = 'relu'))
    
    return model




def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])




if __name__ == "__main__":
    
    install("tensorflow==2.4.1")
    install("numpy==1.21.0")

    args, _ = parse_args()

    print('Training data location: {}'.format(args.train))
    print('Test data location: {}'.format(args.test))
    x_train, y_train = get_train_data(args.train)
    
    x_test, y_test = get_test_data(args.test)

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))


    model = get_model()
    
    
    ##Compile the network 
    ##optimizer = tf.keras.optimizers.SGD(learning_rate)
    #model.compile(loss='mean_squared_error',optimizer='adam')
    ##model.compile(optimizer=optimizer, loss='mse')

    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    print(model.summary())
    
    
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # evaluate on test set
    scores = model.evaluate(x_test, y_test, batch_size, verbose=1)
    print("\nTest MSE :", scores)
    
    model_path = args.sm_model_dir + '/1'
    print("model_path = ", model_path)
       
    # save model
    model.save(args.sm_model_dir + '/1')
    

#     tf.keras.models.save_model(
#       model,
#       os.path.join(args.sm_model_dir, '/1'),
#       overwrite=True,
#       include_optimizer=True
#      )

