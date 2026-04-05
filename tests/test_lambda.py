import json
import pytest
from unittest.mock import patch, MagicMock
import os
from lambda_trigger import lambda_handler

@pytest.fixture
def mock_sagemaker():
    with patch('boto3.client') as mock_client:
        mock_sm = MagicMock()
        mock_client.return_value = mock_sm
        yield mock_sm

def test_lambda_handler_success(mock_sagemaker):
    # Setup mock return value
    mock_sagemaker.start_pipeline_execution.return_value = {
        'PipelineExecutionArn': 'arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline'
    }
    
    event = {"source": "aws.events", "detail-type": "Domain Shift Detected"}
    context = MagicMock()
    
    response = lambda_handler(event, context)
    
    assert response['statusCode'] == 200
    assert "MLOps retraining triggered successfully." in response['body']
    mock_sagemaker.start_pipeline_execution.assert_called_once()

def test_lambda_handler_failure(mock_sagemaker):
    # Setup mock exception
    mock_sagemaker.start_pipeline_execution.side_effect = Exception("S3 bucket not found")
    
    event = {"source": "aws.events"}
    context = MagicMock()
    
    with pytest.raises(Exception) as excinfo:
        lambda_handler(event, context)
        
    assert "S3 bucket not found" in str(excinfo.value)
