import os
from qdrant_client import QdrantClient, models
from typing import List
import httpx
import asyncio
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import plotly.graph_objects as go

# Configuration
PDF_PATH = "Suivi livrables DEV-IA - Comparatif Projet(s) Vs Référentiel.pdf"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "pdf_document_embeddings" # Changed for specificity
EMBEDDING_API_URL = "http://localhost:8001/v1/embeddings" # URL of the Dockerized embedding API
EMBEDDING_MODEL = "google/embeddinggemma-300m" # Model name expected by the API

async def generate_embeddings(texts: List[str], model_name: str) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using the Dockerized embedding API.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            EMBEDDING_API_URL,
            json={"model": model_name, "input": texts},
            timeout=None # Allow long timeouts for large documents
        )
        response.raise_for_status()
        embeddings_data = response.json()["data"]
        return [item["embedding"] for item in embeddings_data]

def store_embeddings_in_qdrant(
    texts: List[str],
    embeddings: List[List[float]],
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int
):
    """
    Stores the generated embeddings and their corresponding texts in Qdrant.
    """
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Determine vector size from the first embedding
    if not embeddings:
        print("No embeddings to store. Exiting.")
        return
    vector_size = 768 # As per user feedback, the model dimension is 768
    print(f"DEBUG: Using fixed vector_size for Qdrant: {vector_size}")

    # Create collection if it doesn't exist
    if client.collection_exists(collection_name=collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting and recreating...")
        client.delete_collection(collection_name=collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created.")

    # Prepare points for upsertion
    points = []
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        points.append(
            models.PointStruct(
                id=i,
                vector=embedding,
                payload={"text": text}
            )
        )
    
    # Upsert points to the collection
    client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
    )
    print(f"Successfully stored {len(points)} embeddings in collection '{collection_name}'")

def display_qdrant_collection_info(collection_name: str, qdrant_host: str, qdrant_port: int):
    """
    Displays information about the specified Qdrant collection.
    """
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"\n--- Qdrant Collection Info for '{collection_name}' ---")
        
        # Prepare data for table
        data = [
            ["Attribute", "Value"],
            ["Status", collection_info.status],
            ["Vectors Count", client.count(collection_name=collection_name).count],
            ["Segments Count", collection_info.segments_count],
        ]
        
        disk_size = "Not directly available"
        if hasattr(collection_info.config.optimizer_config, 'disk_size_bytes'):
            disk_size = f"{collection_info.config.optimizer_config.disk_size_bytes} bytes"
        data.append(["Disk Size", disk_size])
        data.append(["Config", str(collection_info.config)])

        fig = go.Figure(data=[go.Table(
            header=dict(values=['Attribute', 'Value'],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[[row for row in data[1:]], [row for row in data[1:]]],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.show()
        print(f"--- End Collection Info ---\n")
    except Exception as e:
        print(f"Error retrieving collection info for '{collection_name}': {e}")

def display_collection_contents_in_table(collection_name: str, qdrant_host: str, qdrant_port: int):
    """
    Displays the contents (embedding and text payload) of a Qdrant collection in a table format.
    """
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    try:
        scroll_result, _ = client.scroll(
            collection_name=collection_name,
            limit=100, # Limit to first 100 points for display
            with_payload=True,
            with_vectors=True,
        )

        if not scroll_result:
            print(f"\nNo points found in collection '{collection_name}'.")
            return

        ids = [point.id for point in scroll_result]
        embedding_previews = [str(point.vector[:5]) + "..." if point.vector else "N/A" for point in scroll_result]
        text_payloads = [point.payload.get("text", "N/A") if point.payload else "N/A" for point in scroll_result]
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=['ID', 'Embedding (first 5 elements)', 'Text Payload'],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[ids, embedding_previews, text_payloads],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.show()
        print(f"--- End Collection Contents ---\n")

    except Exception as e:
        print(f"Error retrieving collection contents for '{collection_name}': {e}")

async def main():
    print(f"Starting PDF embedding process for '{PDF_PATH}'...")
    
    # 1. Load and chunk PDF
    print(f"Loading and chunking PDF '{PDF_PATH}'...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Chunked into {len(chunked_documents)} pieces.")
    
    # Extract page content for embedding
    chunk_texts = [doc.page_content for doc in chunked_documents]

    # 2. Generate embeddings
    print(f"Generating embeddings using model '{EMBEDDING_MODEL}' via API '{EMBEDDING_API_URL}'...")
    text_embeddings = await generate_embeddings(chunk_texts, EMBEDDING_MODEL)
    print(f"Generated {len(text_embeddings)} embeddings.")
    if text_embeddings:
        print(f"DEBUG: First embedding sample (first 5 elements): {text_embeddings[:5]}...")
    else:
        print("DEBUG: No embeddings were generated.")

    # 3. Store embeddings in Qdrant
    print(f"Storing embeddings in Qdrant at {QDRANT_HOST}:{QDRANT_PORT} in collection '{COLLECTION_NAME}'...")
    store_embeddings_in_qdrant(chunk_texts, text_embeddings, COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT)
    print("PDF embedding process completed.")

    # 4. Display Qdrant collection info
    display_qdrant_collection_info(COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT)

    # 5. Display Qdrant collection contents
    display_collection_contents_in_table(COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT)

if __name__ == "__main__":
    asyncio.run(main())