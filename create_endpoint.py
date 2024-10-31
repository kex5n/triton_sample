import boto3


aws_region='ap-northeast-3'
instance_type = 'ml.g4dn.xlarge'

container = f"763104351884.dkr.ecr.{aws_region}.amazonaws.com/sagemaker-tritonserver:23.03-py3"

# Boto3のSageMakerクライアントを取得する
sagemaker_client = boto3.client('sagemaker', region_name=aws_region)

# 作成するSageMakerモデルの名前（任意）
model_name = 'pytorch-predictor-onnx-model'

# SageMakerモデルを作成する
create_model_response = sagemaker_client.create_model(
    ModelName = model_name,
    ExecutionRoleArn = '<SageMakerの実行ロールのARN>',
    PrimaryContainer = {
        'Image': container,
        'ModelDataUrl': '<S3にアップロードしたモデルアーティファクトのURL>',
    }
)
