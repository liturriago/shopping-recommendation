import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from opensearchpy import OpenSearch

app = FastAPI(title="MELI RecSys MVP")

# 1. Load model globally for optimized container startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

# Processor handles tokenization, Model handles the forward pass
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)
model.eval() # Set to evaluation mode

# 2. OpenSearch Connection Setup
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST")
OPENSEARCH_USER = os.environ.get("OPENSEARCH_USER")
OPENSEARCH_PASS = os.environ.get("OPENSEARCH_PASS")

if not all([OPENSEARCH_HOST, OPENSEARCH_USER, OPENSEARCH_PASS]):
    print("[WARNING] OpenSearch credentials not fully set in environment.")

client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': 443}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=True,
    verify_certs=True
)

class RecommendationRequest(BaseModel):
    query_text: str
    top_k: int = 5

@app.post("/recommend")
async def get_recommendations(req: RecommendationRequest):
    try:
        # A. Real-time text vectorization
        inputs = processor(text=[req.query_text], return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            # This uses the text_projection layer to output a 512-D tensor
            text_features = model.get_text_features(**inputs)
            
        # L2 Normalization (Required for Cosine Similarity in OpenSearch)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        query_vector = text_features.cpu().numpy()[0].tolist()

        # B. Build OpenSearch Hybrid Query
        os_query = {
            "size": req.top_k,
            "_source": ["article_id", "product_name"], 
            "query": {
                "bool": {
                    "should": [
                        {"knn": {"image_vector": {"vector": query_vector, "k": req.top_k}}},
                        {"match": {"product_name": req.query_text}}
                    ]
                }
            }
        }

        # C. Execute search
        response = client.search(index="hm_products", body=os_query)
        
        # D. Format response
        hits = response['hits']['hits']
        results = [{"article_id": hit["_source"]["article_id"], "score": hit["_score"]} for hit in hits]
        
        return {"query": req.query_text, "recommendations": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))