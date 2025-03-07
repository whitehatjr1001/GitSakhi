from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from knowledge_graph import CodeChunk

@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    embedding_dim: int = 1536  # Default for text-embedding-ada-002
    similarity_threshold: float = 0.7
    max_results: int = 10

class VectorStore:
    def __init__(self, config: VectorDBConfig = None):
        self.config = config or VectorDBConfig()
        self.embeddings: Dict[str, np.ndarray] = {}  # chunk_id -> embedding
        self.chunks: Dict[str, CodeChunk] = {}       # chunk_id -> chunk
        
    def add_chunk(self, chunk: CodeChunk) -> None:
        """Add a chunk and its embedding to the store."""
        if not chunk.embedding:
            raise ValueError("Chunk must have an embedding")
            
        embedding = np.array(chunk.embedding)
        if embedding.shape[0] != self.config.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.config.embedding_dim}, got {embedding.shape[0]}")
            
        self.embeddings[chunk.chunk_id] = embedding
        self.chunks[chunk.chunk_id] = chunk
        
    def delete_chunk(self, chunk_id: str) -> None:
        """Remove a chunk from the store."""
        self.embeddings.pop(chunk_id, None)
        self.chunks.pop(chunk_id, None)
        
    def search(self, query_embedding: List[float], top_k: int = None) -> List[Tuple[str, float]]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_embedding: The query vector
            top_k: Number of results to return (defaults to config.max_results)
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        if not self.embeddings:
            return []
            
        top_k = top_k or self.config.max_results
        query_vec = np.array(query_embedding)
        
        # Calculate cosine similarity with all embeddings
        similarities = []
        for chunk_id, embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_vec, embedding)
            if similarity >= self.config.similarity_threshold:
                similarities.append((chunk_id, similarity))
        
        # Sort by similarity score and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """Retrieve a chunk by its ID."""
        return self.chunks.get(chunk_id)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def batch_add_chunks(self, chunks: List[CodeChunk]) -> None:
        """Add multiple chunks at once."""
        for chunk in chunks:
            self.add_chunk(chunk)
