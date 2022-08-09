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
import sys

!{sys.executable} -m pip install "sagemaker==2.91.1"




import os
import boto3
import sagemaker
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.steps import TrainingStep
import time
############################################################################################

BASE_DIR = os.path.dirname(os.path.realpath(__file__))



def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client




def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags





def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    processing_instance_type="ml.m5.large",
    training_instance_type="ml.m5.large",
):
    """Gets a SageMaker ML Pipeline instance working with on CustomerChurn data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
        
    pipeline_session = get_pipeline_session(region, default_bucket)

    # Parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    training_instance_type = ParameterString(name="TrainingInstanceType",default_value="ml.m5.large")
    model_approval_status = ParameterString(name="ModelApprovalStatus",default_value="PendingManualApproval",  
                                            # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
)
    input_data = ParameterString( name="InputDataUrl",default_value=f"s3://sagemaker-eu-west-2-692736957113/sagemaker/CaliforniaHousingPricesData/data/housing.csv",# Change this to point to the s3 location of your raw input data.
    )
    
    #########################################
    # Processing step for feature engineering
    #########################################
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",  # choose any name
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_process = ProcessingStep(
        name="PreprocessAbaloneData",  # choose any name
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
             
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-data", input_data],
    )

    ##############################################
    # Training step for generating model artifacts
    ##############################################
#     model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AbaloneTrain"
    
        
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",  # we are using the Sagemaker built in xgboost algorithm
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.large",
    )
    

   

    # Where to store the trained model
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/model/"
    #model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AbaloneTrain"
    #model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AbaloneTrain"
    

    hyperparameters = {"epochs": 3 }
    tensorflow_version = "2.4.1"
    python_version = "py37"

    tf2_estimator = TensorFlow(
        source_dir=BASE_DIR,
        entry_point="train.py",
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version=tensorflow_version,
        role=role,
        base_job_name=f"{base_job_prefix}/abalone-train",
        output_path=model_path,
        hyperparameters=hyperparameters,
        py_version=python_version,
    )

    #Use the tf2_estimator in a Sagemaker pipelines ProcessingStep.
    #NOTE how the input to the training job directly references the output of the previous step.
    step_train = TrainingStep(
    name="TrainAbaloneModel",
    estimator=tf2_estimator,
    inputs={
        "train":
        TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,content_type="text/csv",),
            
        "test":
        TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,content_type="text/csv",),
        },
    )

    
    
    #####################################
    # Processing step for evaluation
    #####################################

    from sagemaker.workflow.properties import PropertyFile

    # Create SKLearnProcessor object.
    # The object contains information about what container to use, what instance type etc.
    framework_version = "0.23-1"
    evaluate_model_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name= f"{base_job_prefix}/script-abalone-eval",
        role=role,
    )

    # Create a PropertyFile
    # A PropertyFile is used to be able to reference outputs from a processing step, for instance to use in a condition step.
    # For more information, visit https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    # Use the evaluate_model_processor in a Sagemaker pipelines ProcessingStep.
    step_eval = ProcessingStep(
        name= "EvaluateAbaloneModel",
        processor=evaluate_model_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
                
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
                
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
             
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )
    
    
    #########################################################
    # Register model step that will be conditionally executed
    #########################################################
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    # Register model step that will be conditionally executed
    step_register = RegisterModel(
        name="CustomerChurnRegisterModel",
        estimator=tf2_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    
    


    # Condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(  # You can change the condition here
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.
        ),
        right=0.04,  # You can change the threshold here
    )
    step_cond = ConditionStep(
        name="CheckMSEAbaloneEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
