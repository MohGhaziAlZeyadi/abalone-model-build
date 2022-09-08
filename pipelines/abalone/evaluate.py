
import os
import json
import subprocess
import sys
import numpy as np
import pathlib
import tarfile
import argparse
import logging
from sklearn.metrics import mean_squared_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    
# def model_summary(model):
#     # Iterate over model layers
#     for layer in model.layers:
#         print(layer.name, layer)

#     # firstlayer
#     print(model.layers[0].weights)
#     print(model.layers[0].bias.numpy())
#     print(model.layers[0].bias_initializer)

#     # secondlayer
#     print(model.layers[1].weights)
#     print(model.layers[1].bias.numpy())
#     print(model.layers[1].bias_initializer)

#     # 3rdlayer
#     print(model.layers[2].weights)
#     print(model.layers[2].bias.numpy())
#     print(model.layers[2].bias_initializer)
    
#     # lastlayer
#     print(model.layers[3].weights)
#     print(model.layers[3].bias.numpy())
#     print(model.layers[3].bias_initializer)

#     # firstlayer by name
#     print((model.get_layer("1stlayer").weights))

#     # secondlayer by name
#     print((model.get_layer("2ndlayer").weights))
    
#     # 3rdlayer by name
#     print((model.get_layer("3rdlayer").weights))

#     # lastlayer by name
#     print((model.get_layer("lastlayer").weights))



if __name__ == "__main__":
    
    print("Python version")
    print (sys.version)
    print("Version info.")
    print (sys.version_info)

    install("tensorflow==2.4.1")
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall("./model")
        
        
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import optimizers

    #model_loded = tf.keras.models.load_model("./model/1")
    model_loded = tf.keras.models.load_model("./model/1")
    #model_loded.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    #print(model_loded.summary())
    
    #model_summary(model_loded)
    
    
    test_path = "/opt/ml/processing/test/"
    x_test = np.load(os.path.join(test_path, "x_test.npy"))
    y_test = np.load(os.path.join(test_path, "y_test.npy"))
    
#     print('x test', x_test.shape,'y test', y_test.shape)
    
#     print("type(x_test): ", type(x_test))
#     print("(x_test): ",(x_test))
    
#     print("type(y_test)", type(y_test))
#     print("(y_test)", (y_test))
  

    
#     print("Evalaution Start...")
#     batch_size = 64
#     scores = model_loded.evaluate(x_test, y_test, batch_size, verbose=1)
   
#     print("\nTest MSE :", scores)
    
    logger.info("Performing predictions against test data.")
    predictions = model_loded.predict(x_test)
    
    #print("predictions: ", predictions)

    logger.debug("Calculating mean squared error.")
    mse = mean_squared_error(y_test, predictions)
    std = np.std(y_test - predictions)
    
    
     # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "standard_deviation": "NaN"
            }
        }
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with mse: %f", mse)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    #logger.info("Processing Stage Done!")
    