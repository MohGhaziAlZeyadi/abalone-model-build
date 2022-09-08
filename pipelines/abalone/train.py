import argparse
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import subprocess
import sys

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Softmax
from tensorflow.keras import optimizers
from tensorflow.keras import layers

import json
import pathlib
import tarfile


print(tf. __version__) 
print(np. __version__) 
print("************************************************************")
def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    #parser.add_argument('--validation', type=str, required=False, default=os.environ.get('SM_CHANNEL_VALIDATION')) 

    # model directory
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR')) #  A string that represents the path where the training job writes the model artifacts to. After training, artifacts in this directory are uploaded to S3 for model hosting.

    return parser.parse_known_args()




def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
def get_train_data(train_dir):    
    train_df = pd.read_csv(os.path.join(f"{train_dir}/train_df.csv"))    
    print('train_df shape: ', train_df.shape)
    return train_df


def get_test_data(test_dir):
    test_df = pd.read_csv(os.path.join(f"{test_dir}/test_df.csv"))    
    print('test_df shape: ', test_df.shape)
    return test_df


def get_validation_data(validation_dir):
    validation_df = pd.read_csv(os.path.join(f"{validation_dir}/val_df.csv"))    
    print('validation_df shape: ', validation_df.shape)
    return validation_df


def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)

def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["TAC_Closing_Summing_up"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices((dataframe["Fault Symptom"].values, label_binarized))
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)


def make_model():
    shallow_mlp_model = keras.Sequential(
        [
            layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.25),
            layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.25),
            layers.Dense(lookup.vocabulary_size(), activation="sigmoid"),
        ]  
    )
    return shallow_mlp_model


#varibale used for make_dataset function
max_seqlen = 512
#batch_size = 128
batch_size = 128
padding_token = "<pad>"
auto = tf.data.AUTOTUNE


if __name__ == "__main__":
    
    print("Python version")
    print (sys.version)
    print("Version info.")
    print (sys.version_info)
    
    
    #install("tensorflow==2.6.2")
    install("pickle-mixin")
    install("keras==2.6.0")


    args, _ = parse_args()

    print('Training data location: {}'.format(args.train))
    print('Test data location: {}'.format(args.test))
    print('validation data location: {}'.format(args.validation))
    train_df = get_train_data(args.train)    
    test_df = get_test_data(args.test)    
    val_df =  get_validation_data(args.validation)
    
    
    #Multi-label binarization
    #Now we preprocess our labels using the StringLookup layer.

    terms = tf.ragged.constant(train_df["TAC_Closing_Summing_up"].values)
    print(terms)
    lookup =  tf.keras.layers.StringLookup(output_mode="one_hot")
    lookup.adapt(terms)
    vocab = lookup.get_vocabulary()
    print("Vocabulary:\n")
    print(vocab)
    vocab.sort()
    
    sample_label = train_df["TAC_Closing_Summing_up"].iloc[0]
    print(f"Original label: {sample_label}")

    label_binarized = lookup([sample_label])
    print(f"Label-binarized representation: {(label_binarized)}")
    
    print(train_df.head(10))
    
    
    #Now we can prepare the tf.data.Dataset objects.
    train_dataset = make_dataset(train_df, is_train=True)
    validation_dataset = make_dataset(val_df, is_train=False)
    test_dataset = make_dataset(test_df, is_train=False)
    
    print(train_dataset)
    
    #Dataset preview
    print(train_dataset)
    text_batch, label_batch = next(iter(train_dataset))
    
    for i, text in enumerate(text_batch[:5]):
        label = label_batch[i].numpy()[None, ...]
        print(f"Fault symptom describtion: {text}")
        print(f"TAC Closing Summing up(s): {invert_multi_hot(label[0])}")
        print(" ")

    
    
    #Vectorization
    # Source: https://stackoverflow.com/a/18937309/7636462
    #Before we feed the data to our model, we need to vectorize it (represent it in a numerical form).
    #For that purpose, we will use the TextVectorization layer.
    #It can operate as a part of your main model so that the model is excluded from the core preprocessing logic.
    #This greatly reduces the chances of training / serving skew during inference.

    #We first calculate the number of unique words present in the abstracts.

    vocabulary = set()
    train_df["Fault Symptom"].str.lower().str.split().apply(vocabulary.update)
    vocabulary_size = len(vocabulary)
    print(vocabulary_size)
    
    
    
    text_vectorizer = layers.TextVectorization(max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf")

    # `TextVectorization` layer needs to be adapted as per the vocabulary from our training set.
    with tf.device("/CPU:0"):
        text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

    train_dataset = train_dataset.map(lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto).prefetch(auto)
    validation_dataset = validation_dataset.map(lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto).prefetch(auto)
    test_dataset = test_dataset.map(lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto).prefetch(auto)



    #Save text_vectorizer inorder to be used later to create a model for inference
    import pickle 
    # Vector for word "this"
    print (text_vectorizer("Solenoid"))

    # Pickle the config and weights
    pickle.dump({'config': text_vectorizer.get_config(),
                 'weights': text_vectorizer.get_weights()}
                , open("tv_layer.pkl", "wb"))
    # Later you can unpickle and use 
    # `config` to create object and 
    # `weights` to load the trained weights. 

    print(lookup.vocabulary_size())
    
    
    
    #epochs = 200
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))

    shallow_mlp_model = make_model()
    shallow_mlp_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])

    history = shallow_mlp_model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)
    
    print("Tranning finish")
    
    categorical_acc = shallow_mlp_model.evaluate(test_dataset)
    print("Categorical accuracy: ", categorical_acc)
    #print(f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%.")
    
    
    
    # save model
    shallow_mlp_model.save(args.sm_model_dir+ '/1')
    print("The model saved into:- ", args.sm_model_dir+ '/1')
   
    print("***************Loaded Model*******************")

    model_load = tf.keras.models.load_model(args.sm_model_dir+ '/1')
    print("Loaded Model...")
    print(model_load.summary())
    scores_loaded = model_load.evaluate(test_dataset)
    print("Categorical accuracy: ", scores_loaded)
    #print(f"Categorical accuracy on the test set from loaded model is: {round(scores_loaded * 100, 2)}%.")