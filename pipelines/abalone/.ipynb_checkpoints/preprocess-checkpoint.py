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
import os
import argparse
import logging
import pathlib

import boto3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    

    # read in csv
    df = pd.read_csv(fn, names=columns, header=None)

    # drop the "Phone" feature column
    df = df.drop(["ocean_proximity"], axis=1)
    
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
    Y = df[["medianHouseValue"]]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
    
    np.save(os.path.join(f"{base_dir}/train/", "x_train.npy"), x_train)
    np.save(os.path.join(f"{base_dir}/test/", "x_test.npy"), x_test)
    np.save(os.path.join(f"{base_dir}/train/", "y_train.npy"), y_train)
    np.save(os.path.join(f"{base_dir}/test/", "y_test.npy"), y_test)
#     rawdata_s3_prefix = "{}/data/raw".format(prefix)
#     raw_s3 = sagemaker_session.upload_data(path="./data/raw/", key_prefix=rawdata_s3_prefix)
#     print(raw_s3)
    
    
    
    

    # Split the data
#     train_data, validation_data, test_data = np.split(df.sample(frac=1, random_state=1729),[int(0.7 * len(df)), int(0.9 * len(df))],)

#     pd.DataFrame(train_data).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
#     pd.DataFrame(validation_data).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
#     pd.DataFrame(test_data).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

    
    
    
