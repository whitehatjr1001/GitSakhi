import os
import json
import uuid
from typing import Dict, List
from pathlib import Path
from cloudflare import Cloudflare

class CloudflareVectorize:
    """Interface for Cloudflare Vectorize API using official SDK."""
    
    def __init__(self, index_name: str = "code-index"):
        """Initialize Cloudflare client with credentials from environment variables."""
        self.api_key = os.getenv("CLOUDFLARE_API_KEY")
        self.email = os.getenv("CLOUDFLARE_EMAIL")
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.index_name = index_name
        
        if not all([self.api_key, self.email, self.account_id]):
            raise ValueError("Missing required environment variables: CLOUDFLARE_API_KEY, CLOUDFLARE_EMAIL, CLOUDFLARE_ACCOUNT_ID")
        
        # Initialize Cloudflare client
        self.client = Cloudflare(
            api_email=self.email,
            api_key=self.api_key
        )
        
        # Create metadata directory if it doesn't exist
        self.metadata_dir = Path("data/metadata")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.metadata_dir / f"{index_name}_metadata.json"
        self.metadata_map = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load metadata from file if it exists."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata_map, f, indent=2)
    
    def delete_index(self):
        """Delete the vector index if it exists."""
        try:
            response = self.client.vectorize.indexes.delete(
                index_name=self.index_name,
                account_id=self.account_id
            )
            # Clear metadata when index is deleted
            self.metadata_map = {}
            self._save_metadata()
            return {"success": True, "message": "Index deleted successfully"}
        except Exception as e:
            # Handle both "not found" and "already deleted" cases
            error_msg = str(e).lower()
            if "not found" in error_msg or "deleted" in error_msg:
                return {"success": True, "message": "Index already deleted or does not exist"}
            raise Exception(f"Failed to delete index: {str(e)}")
    
    def create_index(self, dimension: int = 1536):
        """Create a vector index if it doesn't exist."""
        try:
            # Delete existing index if it exists (ignore errors)
            try:
                self.delete_index()
            except Exception:
                pass
            
            # Wait a moment to ensure deletion is processed
            import time
            time.sleep(2)
            
            # Create new index
            response = self.client.vectorize.indexes.create(
                account_id=self.account_id,
                config={
                    "dimensions": dimension,
                    "metric": "cosine",
                },
                name=self.index_name
            )
            return response
        except Exception as e:
            raise Exception(f"Failed to create index: {str(e)}")

    def insert_vectors(self, vectors: List[List[float]], metadata: List[Dict]):
        """Insert vectors with metadata into Cloudflare Vectorize."""
        try:
            # Generate unique IDs and store metadata separately
            vectors_data = []
            for vector in vectors:
                vector_id = str(uuid.uuid4())
                vectors_data.append({
                    "id": vector_id,
                    "values": vector
                })
            
            # Store metadata in local map
            for i, meta in enumerate(metadata):
                self.metadata_map[vectors_data[i]["id"]] = meta
            self._save_metadata()
            
            # Create temporary NDJSON file for vectors
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ndjson') as f:
                for vector in vectors_data:
                    f.write(json.dumps(vector) + '\n')
                temp_file = f.name
            
            try:
                # Read the file content
                with open(temp_file, 'rb') as f:
                    file_content = f.read()
                
                # Insert vectors using direct API call with v2 endpoint
                import requests
                url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/vectorize/v2/indexes/{self.index_name}/upsert"
                headers = {
                    "X-Auth-Email": self.email,
                    "X-Auth-Key": self.api_key,
                    "Content-Type": "application/x-ndjson"
                }
                response = requests.post(url, headers=headers, data=file_content)
                
                if response.status_code != 200:
                    raise Exception(f"Failed to insert vectors: {response.text}")
                
                result = response.json()
                if not result.get("success", False):
                    raise Exception(f"Failed to insert vectors: {result}")
                    
                return {"success": True, "mutation_id": result.get("result", {}).get("mutation_id")}
            finally:
                # Clean up temporary file
                os.unlink(temp_file)
                
        except Exception as e:
            raise Exception(f"Failed to insert vectors: {str(e)}")

    def query_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """Query similar vectors from Cloudflare vector index."""
        try:
            # Query vectors using direct API call with v2 endpoint
            import requests
            url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/vectorize/v2/indexes/{self.index_name}/query"
            headers = {
                "X-Auth-Email": self.email,
                "X-Auth-Key": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "vector": query_vector,
                "top_k": top_k
            }
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                raise Exception(f"Query failed: {response.text}")
            
            result = response.json()
            if not result.get("success", False):
                raise Exception(f"Query failed: {result}")
            
            matches = result.get("result", {}).get("matches", [])
            results = []
            for match in matches:
                vector_id = match.get("id")
                metadata = self.metadata_map.get(vector_id, {})
                results.append({
                    "metadata": metadata,
                    "score": match.get("score", 0.0)
                })
            return results
            
        except Exception as e:
            raise Exception(f"Query failed: {str(e)}")
