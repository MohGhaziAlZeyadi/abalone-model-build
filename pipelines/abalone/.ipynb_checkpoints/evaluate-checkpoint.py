
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
    
    import tensorflow as tf    
    from tensorflow import keras
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Flatten, Dense, Softmax
    from tensorflow.keras import optimizers
    
    
    install("numpy==1.19.2")
    train_path = "/opt/ml/processing/train/"
    x_train = np.load(os.path.join(train_path, "x_train.npy"))
    y_train = np.load(os.path.join(train_path, "y_train.npy"))
    
    test_path = "/opt/ml/processing/test/"
    x_test = np.load(os.path.join(test_path, "x_test.npy"))
    y_test = np.load(os.path.join(test_path, "y_test.npy"))

    batch_size = 64
    epochs = 3
    learning_rate = 0.01
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))


    model = get_model()
    
   
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
     
    