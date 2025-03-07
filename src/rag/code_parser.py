from typing import List, Dict, Optional, Set
import tree_sitter
from pathlib import Path
import os
import requests
import tempfile
import shutil
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class CodeNode:
    """Represents a code node with metadata."""
    type: str          # 'function', 'class', 'method', etc.
    name: str         
    start_point: tuple # (line, column)
    end_point: tuple
    content: str
    language: str
    scope: Optional[str] = None  # namespace/class scope

class LanguageParser:
    """Multi-language code parser using Tree-sitter."""
    
    def __init__(self):
        # Initialize parsers for different languages
        self.parsers = {}
        self.languages_dir = Path("./vendor/tree-sitter-languages")
        self.languages_dir.mkdir(parents=True, exist_ok=True)
        
        # Define language configurations
        self.language_configs = {
            'python': {
                'url': 'https://github.com/tree-sitter/tree-sitter-python',
                'file_extensions': ['.py'],
                'function_query': '''
                    (function_definition
                        name: (identifier) @function.name) @function.def
                    (class_definition
                        name: (identifier) @class.name) @class.def
                    (method_definition
                        name: (identifier) @method.name) @method.def
                ''',
            },
            'javascript': {
                'url': 'https://github.com/tree-sitter/tree-sitter-javascript',
                'file_extensions': ['.js', '.jsx', '.ts', '.tsx'],
                'function_query': '''
                    (function_declaration
                        name: (identifier) @function.name) @function.def
                    (class_declaration
                        name: (identifier) @class.name) @class.def
                    (method_definition
                        name: (property_identifier) @method.name) @method.def
                    (arrow_function
                        name: (identifier) @function.name) @function.def
                ''',
            },
            'java': {
                'url': 'https://github.com/tree-sitter/tree-sitter-java',
                'file_extensions': ['.java'],
                'function_query': '''
                    (method_declaration
                        name: (identifier) @method.name) @method.def
                    (class_declaration
                        name: (identifier) @class.name) @class.def
                ''',
            },
            'go': {
                'url': 'https://github.com/tree-sitter/tree-sitter-go',
                'file_extensions': ['.go'],
                'function_query': '''
                    (function_declaration
                        name: (identifier) @function.name) @function.def
                    (method_declaration
                        name: (field_identifier) @method.name) @method.def
                ''',
            },
            'rust': {
                'url': 'https://github.com/tree-sitter/tree-sitter-rust',
                'file_extensions': ['.rs'],
                'function_query': '''
                    (function_item
                        name: (identifier) @function.name) @function.def
                    (impl_item
                        name: (identifier) @method.name) @method.def
                ''',
            }
        }
        
    @lru_cache(maxsize=128)
    def get_language_for_file(self, file_path: str) -> Optional[str]:
        """Determine programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        for lang, config in self.language_configs.items():
            if ext in config['file_extensions']:
                return lang
        return None
        
    def _get_parser(self, language: str) -> Optional[tree_sitter.Parser]:
        """Get or create parser for a language."""
        if language not in self.parsers:
            try:
                parser = tree_sitter.Parser()
                # Try to load from the languages directory first
                lang_path = self.languages_dir / f"{language}.so"
                
                if not lang_path.exists():
                    # Skip if language not available
                    print(f"Warning: Tree-sitter language {language} not available")
                    return None
                    
                parser.language = tree_sitter.Language(str(lang_path), language)
                self.parsers[language] = parser
            except Exception as e:
                print(f"Error loading Tree-sitter language {language}: {str(e)}")
                return None
                
        return self.parsers.get(language)
        
    def parse_code(self, content: str, language: str) -> List[CodeNode]:
        """Parse code content into structured nodes."""
        parser = self._get_parser(language)
        if not parser:
            return []  # Return empty list if parser not available
            
        try:
            tree = parser.parse(bytes(content, 'utf8'))
            
            # Get language-specific query
            query_str = self.language_configs[language]['function_query']
            query = parser.language.query(query_str)
            
            # Capture matches
            nodes = []
            for match in query.matches(tree.root_node):
                for capture in match:
                    node_type = capture.name.split('.')[0]  # function, class, or method
                    if node_type == 'def':
                        continue
                        
                    code_node = CodeNode(
                        type=node_type,
                        name=capture.node.text.decode('utf8'),
                        start_point=capture.node.start_point,
                        end_point=capture.node.end_point,
                        content=content[capture.node.start_byte:capture.node.end_byte].decode('utf8') if isinstance(content, bytes) else content[capture.node.start_byte:capture.node.end_byte],
                        language=language
                    )
                    nodes.append(code_node)
                    
            return nodes
        except Exception as e:
            print(f"Error parsing {language} code: {str(e)}")
            return []
        
    def extract_imports(self, content: str, language: str) -> Set[str]:
        """Extract import statements and dependencies."""
        parser = self._get_parser(language)
        if not parser:
            return set()
            
        try:
            tree = parser.parse(bytes(content, 'utf8'))
            
            imports = set()
            if language == 'python':
                query_str = '''
                    (import_statement
                        name: (dotted_name) @import)
                    (import_from_statement
                        module_name: (dotted_name) @import)
                '''
            elif language in ['javascript', 'typescript']:
                query_str = '''
                    (import_statement
                        source: (string) @import)
                    (call_expression
                        function: (identifier) @require
                        arguments: (arguments (string) @import))
                '''
            else:
                return imports
                
            query = parser.language.query(query_str)
            for match in query.matches(tree.root_node):
                for capture in match:
                    imports.add(capture.node.text.decode('utf8').strip('"\''))
                    
            return imports
        except Exception as e:
            print(f"Error extracting imports from {language} code: {str(e)}")
            return set()
        
    def extract_references(self, content: str, language: str) -> Set[str]:
        """Extract function calls and variable references."""
        parser = self._get_parser(language)
        if not parser:
            return set()
            
        try:
            tree = parser.parse(bytes(content, 'utf8'))
            
            references = set()
            if language == 'python':
                query_str = '''
                    (call
                        function: (identifier) @call)
                    (attribute
                        attribute: (identifier) @attr)
                '''
            elif language in ['javascript', 'typescript']:
                query_str = '''
                    (call_expression
                        function: (identifier) @call)
                    (member_expression
                        property: (property_identifier) @prop)
                '''
            else:
                return references
                
            query = parser.language.query(query_str)
            for match in query.matches(tree.root_node):
                for capture in match:
                    references.add(capture.node.text.decode('utf8'))
                    
            return references
        except Exception as e:
            print(f"Error extracting references from {language} code: {str(e)}")
            return set()
