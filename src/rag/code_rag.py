from typing import Dict, List
import os
from rich.progress import Progress, SpinnerColumn, TextColumn

from .embeddings import OpenAIEmbeddings
from .cloudflare_vectorize import CloudflareVectorize
from .knowledge_graph import KnowledgeGraph

class CodeRAG:
    """Code Retrieval Augmented Generation system."""
    
    def __init__(self, index_name: str = "code-index"):
        """Initialize components."""
        self.embeddings = OpenAIEmbeddings()
        self.vectorize = CloudflareVectorize(index_name=index_name)
        self.knowledge_graph = KnowledgeGraph()
        
    async def index_content(self, repo_structure: List[Dict], file_contents: Dict[str, str]):
        """
        Index content into vector store and knowledge graph.
        
        Args:
            repo_structure: List of files and directories with metadata
            file_contents: Dictionary of file paths to their contents
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            # Build knowledge graph from repo structure
            task = progress.add_task(description="Building knowledge graph")
            self.knowledge_graph.process_repo_structure(repo_structure)
            
            # Process code chunks
            chunks = []
            for item in repo_structure:
                if item['type'] == 'file':
                    content = file_contents.get(item['path'])
                    if content:
                        # Add file as a chunk
                        chunks.append({
                            'type': 'file',
                            'file_path': item['path'],
                            'content': content,
                            'metadata': {
                                'id': item['path'],
                                'type': 'file',
                                'file_path': item['path']
                            }
                        })
                        
                        # Try to extract functions and classes
                        lines = content.split('\n')
                        in_function = False
                        in_class = False
                        current_chunk = []
                        chunk_start = 0
                        
                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            
                            # Check for function definition
                            if stripped.startswith('def '):
                                if current_chunk:
                                    chunk = {
                                        'type': 'function' if in_function else 'class_method' if in_class else 'code',
                                        'file_path': item['path'],
                                        'content': '\n'.join(current_chunk),
                                        'start_line': chunk_start,
                                        'end_line': i - 1,
                                        'metadata': {
                                            'id': f"{item['path']}:{chunk_start}-{i-1}",
                                            'type': 'function' if in_function else 'class_method' if in_class else 'code',
                                            'file_path': item['path'],
                                            'start_line': chunk_start,
                                            'end_line': i - 1
                                        }
                                    }
                                    chunks.append(chunk)
                                current_chunk = [line]
                                chunk_start = i
                                in_function = True
                                
                            # Check for class definition
                            elif stripped.startswith('class '):
                                if current_chunk:
                                    chunk = {
                                        'type': 'function' if in_function else 'class_method' if in_class else 'code',
                                        'file_path': item['path'],
                                        'content': '\n'.join(current_chunk),
                                        'start_line': chunk_start,
                                        'end_line': i - 1,
                                        'metadata': {
                                            'id': f"{item['path']}:{chunk_start}-{i-1}",
                                            'type': 'function' if in_function else 'class_method' if in_class else 'code',
                                            'file_path': item['path'],
                                            'start_line': chunk_start,
                                            'end_line': i - 1
                                        }
                                    }
                                    chunks.append(chunk)
                                current_chunk = [line]
                                chunk_start = i
                                in_class = True
                                in_function = False
                                
                            # Add line to current chunk
                            else:
                                current_chunk.append(line)
                                
                            # Check for end of block
                            if stripped and not line.startswith(' '):
                                if current_chunk:
                                    chunk = {
                                        'type': 'function' if in_function else 'class_method' if in_class else 'code',
                                        'file_path': item['path'],
                                        'content': '\n'.join(current_chunk),
                                        'start_line': chunk_start,
                                        'end_line': i,
                                        'metadata': {
                                            'id': f"{item['path']}:{chunk_start}-{i}",
                                            'type': 'function' if in_function else 'class_method' if in_class else 'code',
                                            'file_path': item['path'],
                                            'start_line': chunk_start,
                                            'end_line': i
                                        }
                                    }
                                    chunks.append(chunk)
                                current_chunk = []
                                in_function = False
                                in_class = False
                        
                        # Add final chunk
                        if current_chunk:
                            chunk = {
                                'type': 'function' if in_function else 'class_method' if in_class else 'code',
                                'file_path': item['path'],
                                'content': '\n'.join(current_chunk),
                                'start_line': chunk_start,
                                'end_line': len(lines) - 1,
                                'metadata': {
                                    'id': f"{item['path']}:{chunk_start}-{len(lines)-1}",
                                    'type': 'function' if in_function else 'class_method' if in_class else 'code',
                                    'file_path': item['path'],
                                    'start_line': chunk_start,
                                    'end_line': len(lines) - 1
                                }
                            }
                            chunks.append(chunk)
            
            # Store chunks in knowledge graph
            self.knowledge_graph.process_code_chunks(chunks)
            progress.update(task, completed=True)
            
            # Create vector index
            task = progress.add_task(description="Creating vector index")
            self.vectorize.create_index(dimension=1536)  # text-embedding-3-small dimension
            progress.update(task, completed=True)
            
            # Generate embeddings and store vectors
            task = progress.add_task(description="Generating embeddings")
            texts = [chunk['content'] for chunk in chunks]
            embeddings = await self.embeddings.embed_texts_async(texts)
            progress.update(task, completed=True)
            
            # Store vectors with metadata
            task = progress.add_task(description="Storing vectors")
            metadata = [chunk['metadata'] for chunk in chunks]
            self.vectorize.insert_vectors(embeddings, metadata)
            progress.update(task, completed=True)
            
    async def query(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query using hybrid search approach.
        
        1. First, find relevant files based on metadata
        2. Then, use vector search for semantic similarity
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of relevant code chunks with metadata
        """
        # Generate query embedding
        query_embedding = (await self.embeddings.embed_texts_async([query]))[0]
        
        # Query vector store
        results = self.vectorize.query_vectors(query_embedding, top_k=top_k)
        
        # Enhance results with related files
        enhanced_results = []
        for result in results:
            # Get related files from knowledge graph
            file_path = result['metadata']['file_path']
            related_files = self.knowledge_graph.get_related_files(file_path)
            
            # Get chunks from related files
            related_chunks = []
            for related_file in related_files:
                related_chunks.extend(self.knowledge_graph.get_file_chunks(related_file))
            
            # Add related information to result
            result['related_files'] = related_files
            result['related_chunks'] = related_chunks
            enhanced_results.append(result)
            
        return enhanced_results
