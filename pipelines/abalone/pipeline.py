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
"""Example workflow pipeline script for CustomerChurn pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import time
import boto3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sagemaker
from sagemaker import get_execution_role

sess = boto3.Session()
sm = sess.client("sagemaker")
role = get_execution_role()
sagemaker_session = sagemaker.Session(boto_session=sess)
bucket = sagemaker_session.default_bucket()
region = boto3.Session().region_name
model_package_group_name = "TF2-California-Housing"  # Model name in model registry
prefix = "tf2-california-housing-pipelines"
pipeline_name = "TF2CaliforniaHousingPipeline"  # SageMaker Pipeline name
current_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())


data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)

raw_dir = os.path.join(os.getcwd(), "data/raw")
os.makedirs(raw_dir, exist_ok=True)

# !aws s3 cp s3://sagemaker-sample-files/datasets/tabular/california_housing/cal_housing.tgz .
# !tar -zxf cal_housing.tgz

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
    "ocean_proximity",
]
cal_housing_df = pd.read_csv("./Data/cal_housing.data", names=columns, header=None)

print(cal_housing_df.head())

X = cal_housing_df[
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
Y = cal_housing_df[["medianHouseValue"]] / 100000

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

np.save(os.path.join(raw_dir, "x_train.npy"), x_train)
np.save(os.path.join(raw_dir, "x_test.npy"), x_test)
np.save(os.path.join(raw_dir, "y_train.npy"), y_train)
np.save(os.path.join(raw_dir, "y_test.npy"), y_test)
rawdata_s3_prefix = "{}/data/raw".format(prefix)
raw_s3 = sagemaker_session.upload_data(path="./data/raw/", key_prefix=rawdata_s3_prefix)
print(raw_s3)


from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat

# raw input data
input_data = ParameterString(name="InputData", default_value=raw_s3)

# training step parameters
training_epochs = ParameterString(name="TrainingEpochs", default_value="100")

# model performance step parameters
accuracy_mse_threshold = ParameterFloat(name="AccuracyMseThreshold", default_value=0.75)

# Inference step parameters
endpoint_instance_type = ParameterString(name="EndpointInstanceType", default_value="ml.m5.large")


#Processing Step
#The first step in the pipeline will preprocess the data to prepare it for training. We create a SKLearnProcessor object similar to the one above, but now parameterized, so we can separately track and change the job configuration as needed, for example to increase the instance type size and count to accommodate a growing dataset.

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

framework_version = "0.23-1"

# Create SKlearn processor object,
# The object contains information about what instance type to use, the IAM role to use etc.
# A managed processor comes with a preconfigured container, so only specifying version is required.
sklearn_processor = SKLearnProcessor(
    framework_version=framework_version,
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name="tf2-california-housing-processing-job",
)

# Use the sklearn_processor in a Sagemaker pipelines ProcessingStep
step_preprocess_data = ProcessingStep(
    name="Preprocess-California-Housing-Data",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
    ],
    code="preprocess.py",
)



#Train model step
#In the second step, the train and validation output from the precious processing step are used to train a model.

from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
import time

# Where to store the trained model
model_path = f"s3://{bucket}/{prefix}/model/"

hyperparameters = {"epochs": training_epochs}
tensorflow_version = "2.4.1"
python_version = "py37"

tf2_estimator = TensorFlow(
    source_dir="code",
    entry_point="train.py",
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version=tensorflow_version,
    role=role,
    base_job_name="tf2-california-housing-train",
    output_path=model_path,
    hyperparameters=hyperparameters,
    py_version=python_version,
)

# Use the tf2_estimator in a Sagemaker pipelines ProcessingStep.
# NOTE how the input to the training job directly references the output of the previous step.
step_train_model = TrainingStep(
    name="Train-California-Housing-Model",
    estimator=tf2_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "test": TrainingInput(
            s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
    },
)
    


#Evaluate model step
#When a model is trained, it's common to evaluate the model on unseen data before registering it with the model registry. This ensures the model registry isn't cluttered with poorly performing model versions. To evaluate the model, create a ScriptProcessor object and use it in a ProcessingStep.

#Note that a separate preprocessed test dataset is used to evaluate the model, and not the output of the processing step. This is only for demo purposes, to ensure the second run of the pipeline creates a model with better performance. In a real-world scenario, the test output of the processing step would be used.

from sagemaker.workflow.properties import PropertyFile

# Create SKLearnProcessor object.
# The object contains information about what container to use, what instance type etc.
evaluate_model_processor = SKLearnProcessor(
    framework_version=framework_version,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name="tf2-california-housing-evaluate",
    role=role,
)

# Create a PropertyFile
# A PropertyFile is used to be able to reference outputs from a processing step, for instance to use in a condition step.
# For more information, visit https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
evaluation_report = PropertyFile(
    name="EvaluationReport", output_name="evaluation", path="evaluation.json"
)

# Use the evaluate_model_processor in a Sagemaker pipelines ProcessingStep.
step_evaluate_model = ProcessingStep(
    name="Evaluate-California-Housing-Model",
    processor=evaluate_model_processor,
    inputs=[
        ProcessingInput(
            source=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
    ],
    code="evaluate.py",
    property_files=[evaluation_report],
)



#Register model step
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel

# Create ModelMetrics object using the evaluation report from the evaluation step
# A ModelMetrics object contains metrics captured from a model.
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=evaluation_s3_uri,
        content_type="application/json",
    )
)

# Create a RegisterModel step, which registers the model with Sagemaker Model Registry.
step_register_model = RegisterModel(
    name="Register-California-Housing-Model",
    estimator=tf2_estimator,
    model_data=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large", "ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name=model_package_group_name,
    model_metrics=model_metrics,
)


#Create the model¶
#The model is created and the name of the model is provided to the Lambda function for deployment. The CreateModelStep dynamically assigns a name to the model.

from sagemaker.workflow.step_collections import CreateModelStep
from sagemaker.tensorflow.model import TensorFlowModel

model = TensorFlowModel(
    role=role,
    model_data=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
    framework_version=tensorflow_version,
    sagemaker_session=sagemaker_session,
)

step_create_model = CreateModelStep(
    name="Create-California-Housing-Model",
    model=model,
    inputs=sagemaker.inputs.CreateModelInput(instance_type="ml.m5.large"),
)


from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

# Create accuracy condition to ensure the model meets performance requirements.
# Models with a test accuracy lower than the condition will not be registered with the model registry.
cond_lte = ConditionLessThanOrEqualTo(
    left=JsonGet(
        step_name=step_evaluate_model.name,
        property_file=evaluation_report,
        json_path="regression_metrics.mse.value",
    ),
    right=accuracy_mse_threshold,
)

# Create a Sagemaker Pipelines ConditionStep, using the condition above.
# Enter the steps to perform if the condition returns True / False.
step_cond = ConditionStep(
    name="MSE-Lower-Than-Threshold-Condition",
    conditions=[cond_lte],
    if_steps=[step_register_model, step_create_model, step_lower_mse_deploy_model_lambda],
    else_steps=[step_higher_mse_send_email_lambda],
)


from sagemaker.workflow.pipeline import Pipeline

# Create a Sagemaker Pipeline.
# Each parameter for the pipeline must be set as a parameter explicitly when the pipeline is created.
# Also pass in each of the steps created above.
# Note that the order of execution is determined from each step's dependencies on other steps,
# not on the order they are passed in below.
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        input_data,
        training_epochs,
        accuracy_mse_threshold,
        endpoint_instance_type,
    ],
    steps=[step_preprocess_data, step_train_model, step_evaluate_model, step_cond],
)



import json

definition = json.loads(pipeline.definition())
print(definition)
pipeline.upsert(role_arn=role)
execution = pipeline.start()









