from typing import Dict, List
import re

class Indexer:
    def __init__(self):
        self.directory_structure = []
        self.file_contents = {}
        
    def parse_content(self, content: str) -> None:
        """Parse the repository content and extract directory structure and file contents."""
        # Split content into sections based on the file headers
        sections = content.split('--------------------------------------------------------------------------------')
        
        current_file = None
        current_content = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Check if this section is a file header
            if section.endswith(':'):
                # Save previous file if exists
                if current_file and current_content:
                    self.file_contents[current_file] = '\n'.join(current_content)
                    current_content = []
                
                # Get new file name
                current_file = section[:-1].strip()
                
            # If this is a directory structure section
            elif '├──' in section:
                self.parse_directory_structure(section)
                
            # Otherwise, this is file content
            elif current_file:
                current_content.append(section)
        
        # Save the last file
        if current_file and current_content:
            self.file_contents[current_file] = '\n'.join(current_content)
    
    def parse_directory_structure(self, structure: str) -> None:
        """Parse the directory structure section."""
        lines = structure.split('\n')
        for line in lines:
            if not line.strip():
                continue
            
            # Count the indentation level
            indent_count = len(re.match(r'\s*', line).group())
            level = indent_count // 4  # 4 spaces per level
            
            # Extract the file/directory name
            name = line.strip().replace('├── ', '').replace('└── ', '').replace('│   ', '')
            
            # Add to structure
            self.directory_structure.append({
                'name': name,
                'level': level,
                'type': 'directory' if '/' in name or name.endswith('/') else 'file',
                'path': name
            })
    
    def get_chunks(self) -> Dict[str, str]:
        """
        Get repository content as named chunks.
        
        Returns:
        Dict[str, str] where:
            - 'directory_structure': The root chunk containing the directory tree
            - 'file_[filepath]': Individual file chunks named after their paths
        """
        chunks = {
            'directory_structure': '\n'.join(
                '  ' * item['level'] + item['name'] + 
                (' (directory)' if item['type'] == 'directory' else ' (file)')
                for item in self.directory_structure
            )
        }
        
        # Add file chunks
        for filepath, content in self.file_contents.items():
            # Clean filepath for chunk name (remove leading slash and special chars)
            clean_path = filepath.lstrip('/').replace('/', '_').replace('.', '_')
            chunks[f'file_{clean_path}'] = content
            
        return chunks
    
    def get_file_content(self, file_path: str) -> str:
        """Get the content of a specific file."""
        return self.file_contents.get(file_path, '')
    
    def get_files_at_level(self, level: int) -> List[Dict]:
        """Get all files/directories at a specific level in the structure."""
        return [item for item in self.directory_structure if item['level'] == level]
    
    def get_structure(self) -> List[Dict]:
        """Get the complete directory structure."""
        return self.directory_structure
    
    def get_all_files(self) -> Dict[str, str]:
        """Get all file contents."""
        return self.file_contents

# Example usage
if __name__ == "__main__":
    from repo_fetcher import RepoFetcher
    
    # Fetch repository content
    url = "https://github.com/whitehatjr1001/ContentAI"
    fetcher = RepoFetcher(url)
    repo_content = fetcher.fetch_repo_content()
    
    if repo_content:
        # Create indexer and parse content
        indexer = Indexer()
        indexer.parse_content(repo_content)
        
        # Get and print chunks
        chunks = indexer.get_chunks()
        
        print("\nDirectory Structure Chunk:")
        print(chunks['directory_structure'])
        
        print("\nFile Chunks:")
        for name, content in chunks.items():
            if name.startswith('file_'):
                print(f"\nChunk: {name}")
                print(content[:200] + "..." if len(content) > 200 else content)
