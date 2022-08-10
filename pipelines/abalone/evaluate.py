
import os
import json
import subprocess
import sys
import numpy as np
import pathlib
import tarfile
import argparse



def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    
def model_summary(model):
    # Iterate over model layers
    for layer in model.layers:
        print(layer.name, layer)

    # firstlayer
    print(model.layers[0].weights)
    print(model.layers[0].bias.numpy())
    print(model.layers[0].bias_initializer)

    # secondlayer
    print(model.layers[1].weights)
    print(model.layers[1].bias.numpy())
    print(model.layers[1].bias_initializer)

    # 3rdlayer
    print(model.layers[2].weights)
    print(model.layers[2].bias.numpy())
    print(model.layers[2].bias_initializer)
    
    # lastlayer
    print(model.layers[3].weights)
    print(model.layers[3].bias.numpy())
    print(model.layers[3].bias_initializer)

    # firstlayer by name
    print((model.get_layer("1stlayer").weights))

    # secondlayer by name
    print((model.get_layer("2ndlayer").weights))
    
    # 3rdlayer by name
    print((model.get_layer("3rdlayer").weights))

    # lastlayer by name
    print((model.get_layer("lastlayer").weights))

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
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR')) #  A string that represents the path where the training job writes the model artifacts to. After training, artifacts in this directory are uploaded to S3 for model hosting.

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


def get_model():
    model = Sequential()
    #Input Layer
    model.add(Dense(8, activation='relu', input_dim = 8, name="1stlayer"))

    #Hidden Layer
    model.add(Dense(64,kernel_initializer='normal', activation='relu', name="2ndlayer"))
    model.add(Dense(32,kernel_initializer='normal', activation='relu', name="3rdlayer"))
    #Output Layer
    model.add(Dense(1,kernel_initializer='normal', activation = 'relu',  name="lastlayer"))
    
    return model



if __name__ == "__main__":

    install("tensorflow==2.4.1")
    
    
    install("numpy==1.19.2")
    

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
   
    
    
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # evaluate on test set
    scores = model.evaluate(x_test, y_test, batch_size, verbose=1)
    print("\nTest MSE :", scores)

    # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "regression_metrics": {
            "mse": {"value": scores, "standard_deviation": "NaN"},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
     
    