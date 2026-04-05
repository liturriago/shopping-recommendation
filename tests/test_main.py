import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import torch
import numpy as np

# We'll mock the expensive imports and model loading before importing the app
with patch('transformers.CLIPProcessor.from_pretrained'), \
     patch('transformers.CLIPModel.from_pretrained'), \
     patch('opensearchpy.OpenSearch'):
    from main import app

client = TestClient(app)

@pytest.fixture
def mock_model():
    with patch('main.model') as mock_m:
        # Mock get_text_features output
        mock_features = torch.randn(1, 512)
        mock_m.get_text_features.return_value = mock_features
        yield mock_m

@pytest.fixture
def mock_opensearch():
    with patch('main.client') as mock_os:
        # Mock search response
        mock_os.search.return_value = {
            'hits': {
                'hits': [
                    {'_source': {'article_id': '123', 'product_name': 'Test Item 1'}, '_score': 0.99},
                    {'_source': {'article_id': '456', 'product_name': 'Test Item 2'}, '_score': 0.88}
                ]
            }
        }
        yield mock_os

def test_read_root():
    # FastAPI doesn't have a / route, but we can check docs or similar
    response = client.get("/docs")
    assert response.status_code == 200

def test_recommend_endpoint(mock_model, mock_opensearch):
    response = client.post(
        "/recommend",
        json={"query_text": "striped shirt", "top_k": 2}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert data["query"] == "striped shirt"
    assert "recommendations" in data
    assert len(data["recommendations"]) == 2
    assert data["recommendations"][0]["article_id"] == "123"

def test_recommend_invalid_input():
    # Test missing required field
    response = client.post("/recommend", json={"top_k": 5})
    assert response.status_code == 422
