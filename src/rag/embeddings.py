from typing import List
import os
from openai import OpenAI, AsyncOpenAI

class OpenAIEmbeddings:
    """OpenAI embeddings using text-embedding-3-small model."""
    
    def __init__(self):
        """Initialize OpenAI client with API key from environment."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
            
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-small"
        
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for texts in batches."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(embeddings)
            
        return all_embeddings
        
    async def embed_texts_async(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for texts in batches asynchronously."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=batch
            )
            embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(embeddings)
            
        return all_embeddings
