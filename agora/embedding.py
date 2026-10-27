import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """Base class for embedding models."""
    
    def encode(self, texts):
        """Encode texts into embeddings."""
        raise NotImplementedError("Must be implemented by subclasses") 
    
    def needs_training(self):
        """Indicate if the model requires training."""
        return False

class SentenceTransformerModel(EmbeddingModel):
    """Wrapper for sentence-transformers models."""
    
    def __init__(self, model_name = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts):
        """Encode texts using sentence-transformers."""
        return self.model.encode(texts)