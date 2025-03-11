import os
import requests
from typing import Dict, List

class CloudflareVectorize:
    """Interface for Cloudflare Vectorized API to manage a vector database."""
    
    def __init__(self, index_name: str = "code-index"):
        """Initialize Cloudflare API with credentials from environment variables."""
        self.api_key = os.getenv("CLOUDFLARE_API_KEY")
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.index_name = index_name
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/vectorize/namespaces/{self.index_name}"
        
        if not all([self.api_key, self.account_id]):
            raise ValueError("Missing required environment variables: CLOUDFLARE_API_KEY, CLOUDFLARE_ACCOUNT_ID")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def create_index(self, dimension: int = 1536):
        """Create a vector index if it doesn't exist."""
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/vectorize/namespaces"
        payload = {
            "name": self.index_name,
            "config": {
                "dimensions": dimension,
                "metric": "cosine"
            }
        }
        response = requests.post(url, headers=self.headers, json=payload)
        return response.json()

    def insert_vectors(self, vectors: List[List[float]], metadata: List[Dict]):
        """Insert vectors with metadata into Cloudflare Vectorized API."""
        url = f"{self.base_url}/vectors"
        payload = {"vectors": []}
        
        for i, vector in enumerate(vectors):
            payload["vectors"].append({
                "id": f"vector_{i}",
                "vector": vector,
                "metadata": metadata[i]
            })
        
        response = requests.put(url, headers=self.headers, json=payload)
        return response.json()

    def query_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """Query the vector database to retrieve similar items."""
        url = f"{self.base_url}/search"
        payload = {
            "vector": query_vector,
            "top_k": top_k
        }
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Query failed: {response.text}")
        
        return response.json().get("result", [])
