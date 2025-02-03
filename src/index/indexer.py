from typing import Dict, List, Optional
import re

class Indexer:
    def __init__(self):
        self.repo_structure = []  # List to maintain order of structure
        self.file_contents = {}   # Dictionary to store file contents
        
    def parse_content(self, content: str) -> None:
        """Parse repository content using the structure as a guide."""
        # Split into sections by the boundary marker
        sections = content.split('--------------------------------------------------------------------------------')
        
        # First, find and parse the repository structure
        for section in sections:
            if '├──' in section or '└──' in section:
                self._parse_repo_structure(section.strip())
                break
        
        # Now parse file contents using the structure as a guide
        current_file = None
        current_content = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Check if this is a file header (ends with ':')
            if section.endswith(':'):
                # Save previous file content if exists
                if current_file and current_content:
                    self.file_contents[current_file] = self._clean_content('\n'.join(current_content))
                    current_content = []
                
                # Extract new file path
                current_file = section[:-1].strip()
                if current_file.startswith('/'):
                    current_file = current_file[1:]  # Remove leading slash
                    
            # If not a header and not structure, it's file content
            elif not ('├──' in section or '└──' in section) and current_file:
                current_content.append(section)
        
        # Save the last file
        if current_file and current_content:
            self.file_contents[current_file] = self._clean_content('\n'.join(current_content))
    
    def _parse_repo_structure(self, structure: str) -> None:
        """Parse the repository structure into a hierarchical list."""
        lines = structure.split('\n')
        current_path = []
        
        for line in lines:
            if not line.strip():
                continue
            
            # Calculate indentation level
            indent = len(re.match(r'\s*', line).group())
            level = indent // 4
            
            # Extract the name (remove tree characters)
            name = line.strip()
            for char in ['├── ', '└── ', '│   ', '    ']:
                name = name.replace(char, '')
            
            # Adjust current path based on level
            current_path = current_path[:level]
            current_path.append(name)
            
            # Create full path
            full_path = '/'.join(current_path)
            
            # Add to structure
            self.repo_structure.append({
                'name': name,
                'level': level,
                'path': full_path,
                'type': 'directory' if any(
                    item['path'].startswith(full_path + '/') 
                    for item in self.repo_structure
                ) else 'file'
            })
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize file content."""
        # Remove line numbers if present
        content = re.sub(r'^\s*\d+\s+\|\s*', '', content, flags=re.MULTILINE)
        return content.strip()
    
    def get_structure_text(self) -> str:
        """Get a formatted text representation of the repository structure."""
        output = []
        for item in self.repo_structure:
            prefix = '  ' * item['level']
            name = item['name']
            if item['type'] == 'directory':
                name += '/'
            output.append(f"{prefix}{name}")
        return '\n'.join(output)
    
    def get_all_chunks(self) -> Dict[str, str]:
        """Get all chunks including structure and file contents."""
        chunks = {
            'structure': self.get_structure_text()
        }
        
        # Add file chunks with proper paths
        for path, content in self.file_contents.items():
            if path in [item['path'] for item in self.repo_structure]:
                chunks[f'file:{path}'] = content
            
        return chunks
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a specific file."""
        return self.file_contents.get(file_path)

# Example usage
if __name__ == "__main__":
    from repo_fetcher import RepoFetcher
    
    url = "https://github.com/whitehatjr1001/ContentAI"
    fetcher = RepoFetcher(url)
    repo_content = fetcher.fetch_repo_content()
    
    if repo_content:
        indexer = Indexer()
        indexer.parse_content(repo_content)
        
        # Print repository structure
        print("\nRepository Structure:")
        print(indexer.get_structure_text())
        
        # Print some file contents
        print("\nExample File Contents:")
        for item in indexer.repo_structure[:2]:  # Show first 2 files
            if item['type'] == 'file':
                content = indexer.get_file_content(item['path'])
                if content:
                    print(f"\n{item['path']}:")
                    print(content[:200] + "..." if len(content) > 200 else content)
