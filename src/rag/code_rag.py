from typing import Dict, List, Optional, Callable
import os
from openai import OpenAI
from ..index.repo_fetcher import RepoFetcher
from ..index.indexer import Indexer
from .code_parser import LanguageParser, CodeNode
from .cloudflare_vectorize import CloudflareVectorize

class CodeChunk:
    def __init__(self, content: str, file_path: str, chunk_type: str = "code", 
                 start_line: int = 0, end_line: int = 0, name: str = "", scope: str = None):
        self.content = content
        self.file_path = file_path
        self.chunk_type = chunk_type
        self.language = self._detect_language(file_path)
        self.embedding = None
        self.start_line = start_line
        self.end_line = end_line
        self.name = name
        self.scope = scope
        
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = file_path.split('.')[-1] if '.' in file_path else ''
        language_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'go': 'go',
            'rs': 'rust',
            'rb': 'ruby',
            'php': 'php',
            'cs': 'csharp',
            'swift': 'swift',
            'kt': 'kotlin',
            'scala': 'scala',
            'md': 'markdown',
            'json': 'json',
            'yaml': 'yaml',
            'yml': 'yaml',
            'html': 'html',
            'css': 'css',
            'sql': 'sql'
        }
        return language_map.get(ext.lower(), 'text')

class CodeRAG:
    """Code Retrieval Augmented Generation system."""
    
    def __init__(self, cloudflare_key: str, cloudflare_email: str, cloudflare_account_id: str, openai_key: str, namespace: str = "code-understanding"):
        """
        Initialize the RAG system.
        
        Args:
            cloudflare_key: Cloudflare API token for authentication
            cloudflare_email: Cloudflare account email for identification
            cloudflare_account_id: Cloudflare account ID for accessing resources
            openai_key: OpenAI API key for embeddings
            namespace: Namespace for vector storage
        """
        self.vectorize = CloudflareVectorize(
            api_key=cloudflare_key,
            email=cloudflare_email,
            account_id=cloudflare_account_id,
            namespace=namespace
        )
        self.openai_client = OpenAI(api_key=openai_key)
        self.code_parser = LanguageParser()
        self.indexer = None
        self.supported_languages = set()  # Track which languages are supported
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI's text-embedding-3-small."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
        
    async def initialize(self):
        """Initialize vector storage."""
        await self.vectorize.create_index(dimension=1536)  # text-embedding-3-small dimension
        
    def _create_chunk_from_node(self, node: CodeNode, file_path: str) -> CodeChunk:
        """Create a code chunk from a Tree-sitter node."""
        return CodeChunk(
            content=node.content,
            file_path=file_path,
            chunk_type=node.type,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            name=node.name,
            scope=node.scope
        )
        
    async def index_repository(self, repo_url: str, progress_callback: Optional[Callable[[str, int, int], None]] = None):
        """
        Index a code repository.
        
        Args:
            repo_url: URL of the repository to index
            progress_callback: Optional callback for progress updates
        """
        if progress_callback:
            progress_callback("Fetching repository", 0, 1)
            
        # Fetch and index repository
        fetcher = RepoFetcher(repo_url)
        repo_content = fetcher.fetch_repo_content()
        if not repo_content:
            raise Exception("Failed to fetch repository content")
            
        self.indexer = Indexer()
        self.indexer.parse_content(repo_content)
        
        if progress_callback:
            progress_callback("Repository fetched", 1, 1)
        
        # Get all chunks including structure and files
        chunks_dict = self.indexer.get_all_chunks()
        
        # Convert to CodeChunk objects
        chunks = []
        
        # Add repository structure as a special chunk
        chunks.append(CodeChunk(
            content=chunks_dict['structure'],
            file_path='repository_structure',
            chunk_type='structure'
        ))
        
        # Process file chunks with Tree-sitter
        total_files = sum(1 for path in chunks_dict if path.startswith('file:'))
        processed_files = 0
        
        for path, content in chunks_dict.items():
            if not path.startswith('file:'):
                continue
                
            file_path = path[5:]  # Remove 'file:' prefix
            language = self.code_parser.get_language_for_file(file_path)
            
            # Add the whole file as a chunk
            file_chunk = CodeChunk(
                content=content,
                file_path=file_path,
                chunk_type='file'
            )
            chunks.append(file_chunk)
            
            # Parse code into semantic chunks if supported language
            if language:
                try:
                    # Extract code nodes (functions, classes, methods)
                    code_nodes = self.code_parser.parse_code(content, language)
                    if code_nodes:  # Language is supported and parsing succeeded
                        self.supported_languages.add(language)
                        for node in code_nodes:
                            chunks.append(self._create_chunk_from_node(node, file_path))
                            
                        # Extract imports and references
                        imports = self.code_parser.extract_imports(content, language)
                        if imports:
                            chunks.append(CodeChunk(
                                content="\n".join(imports),
                                file_path=file_path,
                                chunk_type='imports'
                            ))
                            
                        references = self.code_parser.extract_references(content, language)
                        if references:
                            chunks.append(CodeChunk(
                                content="\n".join(references),
                                file_path=file_path,
                                chunk_type='references'
                            ))
                except Exception as e:
                    print(f"Error parsing {file_path}: {str(e)}")
                    
            processed_files += 1
            if progress_callback:
                progress_callback("Processing chunks", processed_files, total_files)
        
        # Generate embeddings in batches
        batch_size = 100  # Match OpenAI's recommended batch size
        total_chunks = len(chunks)
        processed_chunks = 0
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            
            # Combine content with metadata for better semantic understanding
            texts = []
            for chunk in batch:
                metadata = [
                    f"File: {chunk.file_path}",
                    f"Type: {chunk.chunk_type}",
                    f"Language: {chunk.language}"
                ]
                
                if chunk.name:
                    metadata.append(f"Name: {chunk.name}")
                if chunk.scope:
                    metadata.append(f"Scope: {chunk.scope}")
                if chunk.start_line > 0:
                    metadata.append(f"Lines: {chunk.start_line}-{chunk.end_line}")
                    
                texts.append(f"{chunk.content}\n\n{' | '.join(metadata)}")
            
            try:
                # Get embeddings for batch
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts,
                    encoding_format="float"
                )
                
                # Assign embeddings to chunks
                for chunk, embedding_data in zip(batch, response.data):
                    chunk.embedding = embedding_data.embedding
            except Exception as e:
                print(f"Error generating embeddings for batch: {str(e)}")
                continue  # Skip failed batch
                
            processed_chunks += len(batch)
            if progress_callback:
                progress_callback("Generating embeddings", processed_chunks, total_chunks)
            
        # Store vectors
        if progress_callback:
            progress_callback("Storing vectors", 0, len(chunks))
            
        await self.vectorize.insert_vectors(chunks)
        
        if progress_callback:
            progress_callback("Vectors stored", len(chunks), len(chunks))
            
        # Print language support summary
        if self.supported_languages:
            print("\nLanguage Support:")
            print(f"Full parsing enabled for: {', '.join(sorted(self.supported_languages))}")
            print("Other languages will be processed as plain text")
            
    async def query_repository(
        self,
        query: str,
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Query the repository for relevant code chunks.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of results with chunks and similarity scores
        """
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Search vector store
            results = await self.vectorize.query_vectors(
                query_embedding,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )
            
            return results
        except Exception as e:
            print(f"Error querying repository: {str(e)}")
            return []
