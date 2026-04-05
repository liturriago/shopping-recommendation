import json
import boto3
import os

def lambda_handler(event, context):
    """
    Triggered by EventBridge when a Domain Shift is detected 
    (e.g., changes in the latent space structure due to illumination 
    variations in new catalog images).
    """
    print(f"Alert: Domain Shift detected. Event details: {json.dumps(event)}")
    
    # Initialize SageMaker client
    sagemaker_client = boto3.client('sagemaker')
    pipeline_name = os.environ.get('PIPELINE_NAME', 'meli-domain-adaptation-pipeline')
    
    try:
        # Trigger the pipeline to fine-tune the CLIP model
        response = sagemaker_client.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineExecutionDisplayName="DomainShift-AutoTrigger",
            PipelineParameters=[
                {
                    'Name': 'BaseModel',
                    'Value': 'openai/clip-vit-base-patch32'
                },
                {
                    'Name': 'NewBronzeDataUri',
                    'Value': 's3://meli-recsys-bucket/bronze/latest_drifted_data/'
                }
            ]
        )
        
        print(f"Retraining pipeline started successfully: {response['PipelineExecutionArn']}")
        return {
            'statusCode': 200,
            'body': json.dumps('MLOps retraining triggered successfully.')
        }
        
    except Exception as e:
        print(f"Failed to start pipeline: {str(e)}")
        raise e