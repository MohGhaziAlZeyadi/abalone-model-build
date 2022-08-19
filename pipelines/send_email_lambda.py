
"""
This Lambda function sends an E-Mail to the Data Science team with the MSE from model evaluation step. 
The evaluation.json location in S3 is provided via the `event` argument
"""

import json
import boto3


s3_client = client = boto3.client("s3")


def lambda_handler(event, context):

    print(f"Received Event: {event}")

    evaluation_s3_uri = event["evaluation_s3_uri"]
    path_parts = evaluation_s3_uri.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)

    content = s3_client.get_object(Bucket=bucket, Key=key)
    text = content["Body"].read().decode()
    evaluation_json = json.loads(text)
    mse = evaluation_json["regression_metrics"]["mse"]["value"]

    subject_line = "Please check high MSE ({}) detected on model evaluation".format(mse)
    print(f"Sending E-Mail to Data Science Team with subject line: {subject_line}")

    # TODO - ADD YOUR CODE TO SEND EMAIL...

    return {"statusCode": 200, "body": json.dumps("E-Mail Sent Successfully")}
