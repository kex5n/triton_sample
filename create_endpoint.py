import boto3


aws_region='ap-northeast-3'
instance_type = 'ml.g4dn.xlarge'

container = f"763104351884.dkr.ecr.{aws_region}.amazonaws.com/sagemaker-tritonserver:23.03-py3"

sagemaker_client = boto3.client('sagemaker', region_name=aws_region)

model_name = 'onnx-predictor-model'

create_model_response = sagemaker_client.create_model(
    ModelName = model_name,
    ExecutionRoleArn = 'arn:aws:iam::026896639530:role/sagemaker',
    PrimaryContainer = {
        'Image': container,
        'ModelDataUrl': 's3://test-kex5n-2/model-onnx.tar.gz',
    }
)

endpoint_config_name = 'onnx-predictor-endpoint-config'                         

sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'variant1',
            'ModelName': model_name,
            'InstanceType': instance_type,
            'InitialInstanceCount': 1
        }
    ]
)

endpoint_name = 'onnx-predictor-endpoint'

create_endpoint_response = sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)
