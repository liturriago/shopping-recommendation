![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFCC00?style=for-the-badge&logo=huggingface&logoColor=black)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white)

# AI-Powered Fashion Recommendation System

![RecSys Banner](assets/banner.png)

## Overview
This repository contains an end-to-end Machine Learning Recommendation System designed for high-end fashion retail (inspired by H&M). It leverages **CLIP (Contrastive Language-Image Pre-training)** for multimodal embeddings and **OpenSearch** for high-performance k-NN hybrid search.

The system allows users to search for fashion items using natural language queries, matching them against both image features and textual metadata.

## Core Features
- **Multimodal Search**: Query with text to find visually similar items using CLIP embeddings.
- **k-NN OpenSearch Index**: Efficient vector search in the cloud.
- **MLOps Integration**: Automated retraining pipeline triggered by data drift detection via AWS Lambda.
- **Scalable Ingestion**: Streaming ingestion from S3 using Python generators to minimize memory footprint.

## Architecture

```mermaid
graph TD
    User([User]) -->|Natural Language| API[FastAPI Recommendation service]
    API -->|Embed Query| CLIP[OpenAI CLIP Model]
    CLIP -->|Dense Vector| OS[OpenSearch k-NN Index]
    OS -->|Top K Results| API
    API -->|JSON Response| User

    Ingest[(Ingestion Notebook)] -->|S3| S3[Data Lake]
    Notebook1 -->|Bronze Layer| S3
    S3 -->|Inference| Notebook2[Embeddings Generator]
    Notebook2 -->|Silver Layer| S3
    S3 -->|Bulk Index| Notebook3[OpenSearch Ingest]
    Notebook3 --> OS
    
    EventBridge[EventBridge - Drift Detection] -->|Trigger| Lambda[Lambda Trigger]
    Lambda -->|Start Pipeline| SageMaker[SageMaker Training Pipeline]
    SageMaker -->|Update Model| CLIP
```

## Getting Started

### Prerequisites
- Python 3.11+
- AWS Account (S3, OpenSearch, SageMaker)
- Kaggle API credentials (for dataset ingestion)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/shopping-recommendation.git
   cd shopping-recommendation
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

## Usage

### Running the API
```bash
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`. You can access the interactive docs at `/docs`.

### Notebooks Workflow
1. `ingest_to_s3.ipynb`: Downloads H&M dataset from Kaggle and uploads to S3 Bronze layer.
2. `embeddings_generator.ipynb`: Loads CLIP model, processes images from S3, and generates embeddings (Silver layer).
3. `knn_index_opensearch.ipynb`: Creates the k-NN index in OpenSearch and performs bulk ingestion.

## Testing
We use `pytest` for automated testing.
```bash
pytest
```
Tests are automatically run via GitHub Actions on every push to the `main` branch.

## License
MIT
