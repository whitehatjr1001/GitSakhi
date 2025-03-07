from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch

class CodeEmbedding:
    """Code embedding using Sentence Transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a lightweight but effective model.
        all-MiniLM-L6-v2 is:
        - Free and open source
        - 384 dimension embeddings (good balance of size/performance)
        - Well-suited for code and text
        - Fast inference
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384  # Fixed for all-MiniLM-L6-v2
        
    def embed_text(self, text: str) -> List[float]:
        """Create embedding for text."""
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True)
            return embedding.cpu().numpy().tolist()
            
    def embed_code(self, code: str) -> List[float]:
        """
        Create embedding for code.
        Currently uses same method as text, but could be extended with
        code-specific preprocessing.
        """
        return self.embed_text(code)
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts efficiently."""
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings.cpu().numpy().tolist()
            
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
