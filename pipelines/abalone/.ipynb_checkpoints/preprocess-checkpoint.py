# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Feature engineers the customer churn dataset."""
import argparse
import logging
import pathlib

import boto3
import numpy as np
import pandas as pd

import glob
import os
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    print(input_data)
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/raw-data.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.info("Reading downloaded data.")

    # read in csv
    
    columns = [
    "longitude",
    "latitude",
    "housingMedianAge",
    "totalRooms",
    "totalBedrooms",
    "population",
    "households",
    "medianIncome",
    "medianHouseValue",
    "ocean_proximity"
    ]
    df = pd.read_csv(fn, names=columns, header=None)  
    
    X = df[
    [
        "longitude",
        "latitude",
        "housingMedianAge",
        "totalRooms",
        "totalBedrooms",
        "population",
        "households",
        "medianIncome",
    ]
    ]
    Y = int(df[["medianHouseValue"]] / 100000)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

    np.save(os.path.join(raw_dir, "x_train.npy"), x_train)
    np.save(os.path.join(raw_dir, "x_test.npy"), x_test)
    np.save(os.path.join(raw_dir, "y_train.npy"), y_train)
    np.save(os.path.join(raw_dir, "y_test.npy"), y_test)
#     rawdata_s3_prefix = "{}/data/raw".format(prefix)
#     raw_s3 = sagemaker_session.upload_data(path="./data/raw/", key_prefix=rawdata_s3_prefix)
#     print(raw_s3)
    
    
    
    input_files = glob.glob("{}/*.npy".format("/opt/ml/processing/input"))
    print("\nINPUT FILE LIST: \n{}\n".format(input_files))
    scaler = StandardScaler()
    x_train = np.load(os.path.join("/opt/ml/processing/input", "x_train.npy"))
    scaler.fit(x_train)
    for file in input_files:
        raw = np.load(file)
        # only transform feature columns
        if "y_" not in file:
            transformed = scaler.transform(raw)
        if "train" in file:
            if "y_" in file:
                output_path = os.path.join("/opt/ml/processing/train", "y_train.npy")
                np.save(output_path, raw)
                print("SAVED LABEL TRAINING DATA FILE\n")
            else:
                output_path = os.path.join("/opt/ml/processing/train", "x_train.npy")
                np.save(output_path, transformed)
                print("SAVED TRANSFORMED TRAINING DATA FILE\n")
        else:
            if "y_" in file:
                output_path = os.path.join("/opt/ml/processing/test", "y_test.npy")
                np.save(output_path, raw)
                print("SAVED LABEL TEST DATA FILE\n")
            else:
                output_path = os.path.join("/opt/ml/processing/test", "x_test.npy")
                np.save(output_path, transformed)
                print("SAVED TRANSFORMED TEST DATA FILE\n")

    
