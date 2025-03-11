from typing import Dict, List, Optional
import networkx as nx

class KnowledgeGraph:
    """Manages code relationships as a graph."""
    
    def __init__(self):
        """Initialize empty graph."""
        self.graph = nx.DiGraph()
        self.chunks = {}  # Store chunks by file path
        
    def process_repo_structure(self, repo_structure: List[Dict]):
        """Build graph from repository structure."""
        # Add all nodes first
        for item in repo_structure:
            self.graph.add_node(
                item['path'],
                type=item['type'],
                name=item['name']
            )
            
        # Add edges for parent-child relationships
        for item in repo_structure:
            if '/' in item['path']:
                parent_path = '/'.join(item['path'].split('/')[:-1])
                if parent_path in self.graph:
                    self.graph.add_edge(parent_path, item['path'], relation='contains')
                    
    def process_code_chunks(self, chunks: List[Dict]):
        """Store code chunks by file path."""
        for chunk in chunks:
            file_path = chunk['file_path']
            if file_path not in self.chunks:
                self.chunks[file_path] = []
            self.chunks[file_path].append(chunk)
            
    def get_chunks_for_embedding(self) -> List[Dict]:
        """Get chunks ready for embedding."""
        all_chunks = []
        for chunks in self.chunks.values():
            all_chunks.extend(chunks)
        return all_chunks
        
    def get_file_chunks(self, file_path: str) -> List[Dict]:
        """Get all chunks for a file."""
        return self.chunks.get(file_path, [])
        
    def get_related_files(self, file_path: str) -> List[str]:
        """Get files related to the given file."""
        related = []
        
        # Get parent directory
        if '/' in file_path:
            parent = '/'.join(file_path.split('/')[:-1])
            if parent in self.graph:
                # Get sibling files
                for sibling in self.graph.neighbors(parent):
                    if sibling != file_path:
                        related.append(sibling)
                        
        # Get child files if directory
        if file_path in self.graph:
            related.extend(list(self.graph.neighbors(file_path)))
            
        return related
