
import os
import json
import subprocess
import sys
import numpy as np
import pathlib
import tarfile
import argparse



# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    

    
# def parse_args():

#     parser = argparse.ArgumentParser()

#     # model directory
#     parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

#     return parser.parse_known_args()




# if __name__ == "__main__":

#     install("tensorflow==2.4.1")
#     install("numpy==1.19.2")
    
    
#     import tensorflow as tf
#     from tensorflow import keras
    
#     args, _ = parse_args()

#     #model_path = args.sm_model_dir + '/1'
    
#     print(tf. __version__) 
#     print(np. __version__) 
#     print("************************************************************")
    

# #     model = tf.keras.models.load_model("./model/1")
    
# #     print(model.summary())
    
#     #model.compile(loss='mean_squared_error',optimizer='adam')
    
#     test_path = "/opt/ml/processing/test/"
#     #test_path = "/opt/ml/test/"
#     x_test = np.load(os.path.join(test_path, "x_test.npy"))
#     y_test = np.load(os.path.join(test_path, "y_test.npy"))
#     print('x test', x_test.shape,'y test', y_test.shape)
    
    
#     model_path = "/opt/ml/processing/model/model.tar.gz"
#     with tarfile.open(model_path) as tar:
#         tar.extractall(path=".")
        
#     model_load = tf.keras.models.load_model("/opt/ml/processing/model/model.tar.gz")
#     scores_loaded = model_load.evaluate(x_test, y_test, batch_size, verbose=1)
#     print("\nTest MSE after loading the model :", scores_loaded)
#     # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
#     report_dict = {
#         "regression_metrics": {
#             "mse": {"value": scores, "standard_deviation": "NaN"},
#         },
#     }

#     output_dir = "/opt/ml/processing/evaluation"

    
#     pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

#     evaluation_path = f"{output_dir}/evaluation.json"
#     with open(evaluation_path, "w") as f:
#         f.write(json.dumps(report_dict))

import os
import json
import subprocess
import sys
import numpy as np
import pathlib
import tarfile


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":

    install("tensorflow==2.4.1")
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall("./model")
    import tensorflow as tf

    model = tf.keras.models.load_model("./model/1")
    print(model.summary())
    test_path = "/opt/ml/processing/test/"
    x_test = np.load(os.path.join(test_path, "x_test.npy"))
    y_test = np.load(os.path.join(test_path, "y_test.npy"))
    scores = model.evaluate(x_test, y_test, verbose=2)
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
     
    