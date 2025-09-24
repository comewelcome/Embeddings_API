# Embeddings API

This project provides a Dockerized API for generating embeddings using various pre-trained models from the `sentence-transformers` library. It offers a flexible and scalable solution for integrating embedding generation into your applications, particularly with GPU acceleration.

## Table of Contents

*   [Features](#features)
*   [Prerequisites](#prerequisites)
*   [Getting Started](#getting-started)
    *   [Hugging Face Authentication (for gated models)](#hugging-face-authentication-for-gated-models)
    *   [Build the Docker Image](#1-build-the-docker-image)
    *   [Run the Docker Container](#2-run-the-docker-container)
    *   [Test the API](#3-test-the-api)
*   [API Endpoints](#api-endpoints)
*   [Supported Models](#supported-models)
*   [Development](#development)

## Features

*   **Dockerized API**: Easily deployable and scalable using Docker.
*   **GPU Acceleration**: Leverages NVIDIA GPUs for efficient embedding generation.
*   **Dynamic Model Loading**: Supports loading various `sentence-transformers` models on demand.
*   **OpenAI-like API**: Provides an API interface similar to OpenAI's embeddings endpoint for ease of integration.

## Prerequisites

Before you begin, ensure you have the following installed:

*   [Docker](https://docs.docker.com/get-docker/)
*   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support)

## Getting Started

Follow these steps to set up and run the Embeddings API.

### Hugging Face Authentication (for gated models)

Some models, such as `google/embeddinggemma-300m`, are hosted on gated repositories on Hugging Face and require authentication for download. To access these models, follow these steps:

1.  **Create a Hugging Face Account**: If you don't have one, sign up at [Hugging Face](https://huggingface.co/).
2.  **Accept Model Terms**: Navigate to the specific model page (e.g., `google/embeddinggemma-300m`) and accept its terms and conditions.
3.  **Generate an API Token**: Go to your Hugging Face settings, then "Access Tokens", and generate a new token with "read" access.
4.  **Create a `.env` file**: In the root directory of this project (e.g., `./.env`), create a file named `.env` with the following content, replacing `"hf_YOUR_TOKEN_HERE"` with your newly generated API token:

    ```
    HUGGING_FACE_HUB_TOKEN="hf_YOUR_TOKEN_HERE"
    ```
    The `docker-compose.yml` configuration will automatically load this token from the `.env` file.

### 1. Build the Docker Image

First, build the Docker image for the API. Navigate to the project's root directory in your terminal and execute:

```bash
docker build -t embed-api .
```

### 2. Run the Docker Container

You can run the container with GPU support using `docker-compose`. Ensure your `.env` file is correctly configured if you are using gated models.

```bash
docker compose up -d
```

Alternatively, you can run it directly using `docker run`:

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

#### Example 3: Using `google/embeddinggemma-300m` (Requires Hugging Face Authentication)

```bash
curl -X POST "http://localhost:8001/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{"model":"google/embeddinggemma-300m","input":["Which planet is known as the Red Planet?", "Mars, known for its reddish appearance, is often referred to as the Red Planet."]}'
```

## API Endpoints

*   **POST `/v1/embeddings`**
    *   **Description**: Generates embeddings for the provided input text using a specified model.
    *   **Request Body**:
        ```json
        {
          "model": "model_name",
          "input": ["text1", "text2"]
        }
        ```
        *   `model` (string, required): The name of the `sentence-transformers` model to use (e.g., "sentence-transformers/all-mpnet-base-v2", "google/embeddinggemma-300m").
        *   `input` (array of strings, required): A list of text strings for which to generate embeddings.
    *   **Response**:
        ```json
        {
          "data": [
            {
              "embedding": [...]
            },
            {
              "embedding": [...]
            }
          ]
        }
        ```        *   `data` (array of objects): A list of embedding objects, where each object contains:
            *   `embedding` (array of floats): The generated embedding for the corresponding input text.

## Supported Models

The API supports a wide range of pre-trained models from the `sentence-transformers` library. You can specify any compatible model name in the request body. Popular choices include:

*   `sentence-transformers/all-mpnet-base-v2`
*   `sentence-transformers/multi-qa-mpnet-base-dot-v1`
*   `google/embeddinggemma-300m` (Requires Hugging Face authentication)

## Development

To facilitate local development, you can build the Docker image and run it as described in the "Getting Started" section. Ensure that any necessary ports are exposed for your development environment and that your `.env` file is correctly configured for model access.