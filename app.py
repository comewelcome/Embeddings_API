from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI(title="Embedding API Mimic OpenAI")

# Cache des modèles pour ne pas les recharger à chaque requête
model_cache = {}

# Schéma de la requête OpenAI
class EmbeddingRequest(BaseModel):
    model: str
    input: List[str]

# Schéma de la réponse OpenAI
class EmbeddingData(BaseModel):
    embedding: List[float]

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(req: EmbeddingRequest):
    # Charger le modèle s'il n'est pas déjà en cache
    if req.model not in model_cache:
        model_cache[req.model] = SentenceTransformer(req.model)
    
    # Créer les embeddings
    embeddings = model_cache[req.model].encode(req.input)
    
    # Retourner sous le format OpenAI
    return {
        "data": [{"embedding": emb.tolist()} for emb in embeddings]
    }

# Endpoint test rapide
@app.get("/")
def root():
    return {"status": "API running"}
