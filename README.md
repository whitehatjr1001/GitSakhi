# Code Understanding AI

An intelligent system for analyzing and understanding code repositories using RAG (Retrieval Augmented Generation) and LLM technologies.

## Features

- **Multi-Language Support**: Python, JavaScript, Java, Go, Rust, and more
- **Intelligent Code Parsing**: Accurate extraction of functions, classes, and methods
- **Advanced Code Search**: Semantic search using Cloudflare Vectorize
- **AI-Powered Analysis**: Code explanation and analysis using Groq LLM
- **Rich Context Understanding**: Tracks dependencies and relationships between code components

## Prerequisites

- Python 3.8+
- Cloudflare account with Vectorize enabled
- Groq API access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/code-understanding-ai.git
cd code-understanding-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` and add your API credentials:
- `CLOUDFLARE_ACCOUNT_ID`: Your Cloudflare account ID
- `CLOUDFLARE_API_TOKEN`: Cloudflare API token with Vectorize permissions
- `GROQ_API_KEY`: Your Groq API key

## Usage

1. Index a repository:
```bash
python main.py --repo https://github.com/user/repo
```

2. Query the codebase:
```bash
python main.py --query "How does the authentication system work?"
```

## Example Output

```
Repository Statistics
┌────────────┬──────────────────┐
│ Language   │ Number of Chunks │
├────────────┼──────────────────┤
│ Python     │ 45              │
│ JavaScript │ 23              │
│ Java       │ 12              │
└────────────┴──────────────────┘

Relevant Code Sections:
╭─ Code Chunk ───────────────────────╮
│ File: auth/login.py                │
│ Language: python                   │
│                                    │
│ def authenticate_user(...)         │
╰────────────────────────────────────╯

LLM Analysis:
The authentication system uses JWT tokens...

##Scraping done 
###Issues
indexer is not working properly Done competed 

Tasks pendong 
DB curateing with knowledge graphs 

LLM integration 

## Architecture

1. **Code Parsing**:
   - Tree-sitter for accurate AST parsing
   - Language-specific chunking strategies
   - Dependency tracking

2. **Embedding Pipeline**:
   - Sentence Transformers (all-MiniLM-L6-v2)
   - 384-dimensional embeddings
   - Batch processing support

3. **Vector Storage**:
   - Cloudflare Vectorize for efficient similarity search
   - Metadata storage
   - Relationship tracking

4. **LLM Integration**:
   - Groq's Mixtral-8x7b-32768 model
   - Context-aware code analysis
   - Multi-language support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
