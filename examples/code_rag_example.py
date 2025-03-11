import asyncio
import os
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel
from rich.table import Table

from src.index.repo_fetcher import RepoFetcher
from src.index.indexer import Indexer
from src.rag.code_rag import CodeRAG

async def main():
    # Load environment variables from .env
    load_dotenv()
    
    # Required environment variables
    required_vars = [
        "CLOUDFLARE_API_KEY",
        "CLOUDFLARE_EMAIL", 
        "CLOUDFLARE_ACCOUNT_ID",
        "OPENAI_API_KEY"
    ]
    
    # Check for missing variables
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        print(f"[red]Missing required environment variables: {', '.join(missing)}")
        return
        
    try:
        # Initialize components independently
        repo_url = "https://github.com/whitehatjr1001/ContentAI"
        fetcher = RepoFetcher(url=repo_url)
        indexer = Indexer()
        code_rag = CodeRAG(index_name="example-index")
        
        # Fetch repository content
        print(f"\n[yellow]Fetching repository: {repo_url}")
        repo_content = fetcher.fetch_repo_content()
        if not repo_content:
            print("[red]Failed to fetch repository content")
            return
        
        # Parse content into structure and files
        print("\n[yellow]Parsing repository content")
        indexer.parse_content(repo_content)
        
        # Index content into CodeRAG
        print("\n[yellow]Building knowledge graph and generating embeddings")
        await code_rag.index_content(indexer.repo_structure, indexer.file_contents)
        
        # Example queries
        queries = [
            "What is the main purpose of this repository?",
            "Show me the key features and functionality",
            "Find code related to API endpoints",
            "How is the data processed and stored?"
        ]
        
        # Run queries
        print("\n[green]Running example queries:")
        for query in queries:
            print(f"\n[blue]Query: {query}")
            results = await code_rag.query(query, top_k=3)
            
            # Print results
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                
                # Create result table
                table = Table(title=f"Result {i}", show_header=False)
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")
                
                # Add main result info
                table.add_row("File", metadata['file_path'])
                if metadata.get('start_line') is not None:
                    table.add_row("Lines", f"{metadata['start_line']}-{metadata['end_line']}")
                table.add_row("Type", metadata['type'])
                table.add_row("Score", f"{result['score']:.3f}")
                
                # Add related files
                if result.get('related_files'):
                    table.add_row("Related Files", "\n".join(result['related_files']))
                    
                # Print content snippet
                content_lines = result['content'].split('\n')[:5]  # Show first 5 lines
                if len(content_lines) > 0:
                    content_preview = '\n'.join(content_lines)
                    if len(content_lines) == 5:
                        content_preview += "\n..."
                    print(Panel(content_preview, title="Content Preview"))
                
                print(table)
                print()
                
    except Exception as e:
        print(f"[red]Error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
