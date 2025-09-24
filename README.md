# Embed API

This project provides a Dockerized API for generating embeddings using various pre-trained sentence transformer models.

## Prerequisites

Before you begin, ensure you have the following installed:

*   Docker
*   NVIDIA Container Toolkit (for GPU support)

## Getting Started

Follow these steps to set up and run the Embed API.

### 1. Build the Docker Image

First, build the Docker image for the API:

```bash
docker build -t embed-api .
```

### 2. Run the Docker Container

You can run the container with GPU support using `docker-compose`:

```bash
docker compose up -d```

Alternatively, you can run it directly with `docker run`:

```bash
docker run -d \
  --runtime=nvidia \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=1 \
  -p 8001:8000 \
  --name embed-api \
  embed-api
```

### 3. Test the API

Once the container is running, you can test the API endpoints using `curl`.

#### Example 1: Using `sentence-transformers/all-mpnet-base-v2`

```bash
curl -X POST "http://localhost:8001/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{"model":"sentence-transformers/all-mpnet-base-v2","input":["Bonjour", "Comment ça va ?"]}'
```

#### Example 2: Using `sentence-transformers/multi-qa-mpnet-base-dot-v1`

```bash
curl -X POST "http://localhost:8001/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{"model":"sentence-transformers/multi-qa-mpnet-base-dot-v1","input":["Bonjour", "Comment ça va ?"]}'
```

## API Endpoints

*   **POST `/v1/embeddings`**: Generates embeddings for the provided input text using a specified model.
    *   **Request Body**:
        ```json
        {
          "model": "model_name",
          "input": ["text1", "text2"]
        }
        ```
    *   **Response**:
        ```json
        {
          "embeddings": [
            [...embedding for text1...],
            [...embedding for text2...]
          ]
        }
        ```

## Models

The API supports various models from the `sentence-transformers` library. You can specify the desired model in the request body.

## Development

To develop locally, you can build the Docker image and run it as described above. Make sure to expose the necessary ports for your development environment.