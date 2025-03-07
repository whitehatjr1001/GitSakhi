from typing import Dict, List, Optional, Tuple
import json
from cloudflare import Cloudflare

class CloudflareVectorize:
    """Interface for Cloudflare Vectorize vector database."""
    
    def __init__(self, api_key: str, email: str, account_id: str, namespace: str = "code-understanding"):
        """
        Initialize Cloudflare Vectorize client.
        
        Args:
            api_key: Cloudflare API token for authentication (CLOUDFLARE_API_TOKEN)
            email: Cloudflare account email for identification
            account_id: Cloudflare account ID for accessing resources
            namespace: Namespace for vector storage
        """
        self.client = Cloudflare(
            token=api_key,       # API token for authentication (not api_key)
            email=email          # Email for identification
        )
        self.account_id = account_id
        self.namespace = namespace
        
    async def create_index(self, dimension: int = 1536):
        """
        Create a new vector index if it doesn't exist.
        
        Args:
            dimension: Dimension of vectors (default: 1536 for text-embedding-3-small)
        """
        try:
            # List existing indexes
            indexes = self.client.vectorize.indexes.get(account_id=self.account_id)
            index_exists = any(idx.get('name') == self.namespace for idx in indexes)
            
            if not index_exists:
                # Create new index
                self.client.vectorize.indexes.post(
                    account_id=self.account_id,
                    data={
                        "name": self.namespace,
                        "dimension": dimension,
                        "metric": "cosine",
                        "metadata_fields": [
                            {"name": "file_path", "type": "text"},
                            {"name": "language", "type": "text"},
                            {"name": "chunk_type", "type": "text"},
                            {"name": "content", "type": "text"}
                        ]
                    }
                )
        except Exception as e:
            raise Exception(f"Failed to create/verify index: {str(e)}")
            
    async def insert_vectors(self, chunks: List['CodeChunk']) -> None:
        """
        Insert vectors into the index.
        
        Args:
            chunks: List of CodeChunk objects with embeddings
        """
        # Format vectors for insertion
        vectors = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
                
            vectors.append({
                "id": self._generate_chunk_id(chunk),
                "values": chunk.embedding,
                "metadata": {
                    "file_path": chunk.file_path,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    "content": chunk.content
                }
            })
            
        # Insert in batches of 100 (matching OpenAI's batch size)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.client.vectorize.vectors.post(
                    account_id=self.account_id,
                    index_name=self.namespace,
                    data={"vectors": batch}
                )
            except Exception as e:
                print(f"Error inserting batch {i//batch_size}: {str(e)}")
                continue  # Try next batch
            
    async def query_vectors(
        self,
        query_embedding: List[float],
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Query vectors by similarity.
        
        Args:
            query_embedding: Query vector
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of results with chunks and similarity scores
        """
        try:
            # Query the index
            response = self.client.vectorize.indexes.query(
                account_id=self.account_id,
                index_name=self.namespace,
                vector=query_embedding,
                top_k=max_results
            )
            
            # Format results
            results = []
            for match in response.get("matches", []):
                if match.get("score", 0) < similarity_threshold:
                    continue
                    
                results.append({
                    "chunk": {
                        "content": match["metadata"]["content"],
                        "file_path": match["metadata"]["file_path"],
                        "language": match["metadata"]["language"],
                        "chunk_type": match["metadata"]["chunk_type"]
                    },
                    "score": match["score"]
                })
                
            return results
        except Exception as e:
            print(f"Error querying vectors: {str(e)}")
            return []
        
    def _generate_chunk_id(self, chunk: 'CodeChunk') -> str:
        """Generate a unique ID for a chunk."""
        return f"{chunk.file_path}:{hash(chunk.content)}"
