from typing import Dict, List, Optional, Set
import networkx as nx
from dataclasses import dataclass
from code_parser import LanguageParser, CodeNode

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    chunk_id: str
    repo_url: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str  # 'function', 'class', 'module', etc.
    dependencies: Set[str] = None  # Set of imported module names
    references: Set[str] = None    # Set of referenced chunk_ids
    embedding: List[float] = None  # Vector embedding

class CodeKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.chunks: Dict[str, CodeChunk] = {}
        self.parser = LanguageParser()
        
    def add_chunk(self, chunk: CodeChunk) -> None:
        """Add a code chunk to the knowledge graph."""
        self.chunks[chunk.chunk_id] = chunk
        self.graph.add_node(chunk.chunk_id, 
                          type=chunk.chunk_type,
                          language=chunk.language,
                          file_path=chunk.file_path,
                          repo_url=chunk.repo_url)
        
        # Add edges for dependencies
        if chunk.dependencies:
            for dep in chunk.dependencies:
                self.graph.add_edge(chunk.chunk_id, dep, edge_type='imports')
                
        # Add edges for references
        if chunk.references:
            for ref in chunk.references:
                self.graph.add_edge(chunk.chunk_id, ref, edge_type='references')
    
    def extract_chunks_from_file(self, file_path: str, content: str, repo_url: str) -> List[CodeChunk]:
        """Extract code chunks from a file using language-specific parsing."""
        chunks = []
        
        # Detect language
        language = self.parser.get_language_for_file(file_path)
        if not language:
            # Handle as plain text if language not recognized
            chunk = CodeChunk(
                chunk_id=file_path,
                repo_url=repo_url,
                file_path=file_path,
                content=content,
                start_line=1,
                end_line=len(content.splitlines()),
                language='text',
                chunk_type='file',
                dependencies=set(),
                references=set()
            )
            chunks.append(chunk)
            return chunks
            
        try:
            # Parse code into nodes
            code_nodes = self.parser.parse_code(content, language)
            
            # Extract imports and references
            imports = self.parser.extract_imports(content, language)
            references = self.parser.extract_references(content, language)
            
            # Create chunks from nodes
            for node in code_nodes:
                chunk_id = f"{file_path}::{node.name}"
                chunk = CodeChunk(
                    chunk_id=chunk_id,
                    repo_url=repo_url,
                    file_path=file_path,
                    content=node.content,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    language=language,
                    chunk_type=node.type,
                    dependencies=imports,
                    references=references
                )
                chunks.append(chunk)
                
            # If no chunks were extracted (e.g., data/config files), create a file-level chunk
            if not chunks:
                chunk = CodeChunk(
                    chunk_id=file_path,
                    repo_url=repo_url,
                    file_path=file_path,
                    content=content,
                    start_line=1,
                    end_line=len(content.splitlines()),
                    language=language,
                    chunk_type='file',
                    dependencies=imports,
                    references=references
                )
                chunks.append(chunk)
                
        except Exception as e:
            # Fallback to file-level chunk on parsing error
            chunk = CodeChunk(
                chunk_id=file_path,
                repo_url=repo_url,
                file_path=file_path,
                content=content,
                start_line=1,
                end_line=len(content.splitlines()),
                language=language,
                chunk_type='file',
                dependencies=set(),
                references=set()
            )
            chunks.append(chunk)
            
        return chunks
    
    def get_related_chunks(self, chunk_id: str, max_depth: int = 2) -> List[str]:
        """Get related chunk IDs up to a certain depth in the graph."""
        if chunk_id not in self.graph:
            return []
            
        related = set()
        for depth in range(max_depth):
            neighbors = set(nx.single_source_shortest_path(self.graph, chunk_id, cutoff=depth + 1).keys())
            related.update(neighbors)
            
        related.discard(chunk_id)  # Remove the source chunk
        return list(related)
    
    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """Get the content of a specific chunk."""
        chunk = self.chunks.get(chunk_id)
        return chunk.content if chunk else None
        
    def get_language_stats(self) -> Dict[str, int]:
        """Get statistics about programming languages in the codebase."""
        stats = {}
        for chunk in self.chunks.values():
            stats[chunk.language] = stats.get(chunk.language, 0) + 1
        return stats
