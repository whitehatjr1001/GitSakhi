import asyncio
import argparse
import os
from dotenv import load_dotenv
from src.rag.code_rag import CodeRAG
from src.llm.groq_llm import GroqLLM, CodeContext
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich import print as rprint

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()

async def index_repository(rag: CodeRAG, repo_url: str):
    """Index a repository and show progress."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        # Create tasks for each stage
        master_task = progress.add_task("[cyan]Processing repository...", total=None)
        fetch_task = progress.add_task("[green]Fetching code...", total=1, visible=True)
        chunk_task = progress.add_task("[yellow]Processing chunks...", total=None, visible=False)
        embed_task = progress.add_task("[magenta]Generating embeddings...", total=None, visible=False)
        store_task = progress.add_task("[blue]Storing vectors...", total=None, visible=False)
        
        def update_progress(stage: str, current: int, total: int):
            """Update progress bars based on stage."""
            if stage == "Fetching repository":
                progress.update(fetch_task, completed=0)
            elif stage == "Repository fetched":
                progress.update(fetch_task, completed=1, visible=False)
                progress.update(master_task, description="[cyan]Processing code chunks...")
            elif stage == "Processing chunks":
                if not progress.tasks[chunk_task].visible:
                    progress.update(chunk_task, total=total, visible=True)
                progress.update(chunk_task, completed=current)
            elif stage == "Generating embeddings":
                if not progress.tasks[embed_task].visible:
                    progress.update(chunk_task, visible=False)
                    progress.update(embed_task, total=total, visible=True)
                    progress.update(master_task, description="[cyan]Generating embeddings...")
                progress.update(embed_task, completed=current)
            elif stage == "Storing vectors":
                if not progress.tasks[store_task].visible:
                    progress.update(embed_task, visible=False)
                    progress.update(store_task, total=total, visible=True)
                    progress.update(master_task, description="[cyan]Storing vectors...")
                progress.update(store_task, completed=current)
            elif stage == "Vectors stored":
                progress.update(store_task, completed=total, visible=False)
                progress.update(master_task, description="[cyan]Repository indexed!")
        
        # Initialize vector store and index repository
        await rag.initialize()
        await rag.index_repository(repo_url, progress_callback=update_progress)
        
    # Show repository statistics
    if rag.indexer:
        chunks = rag.indexer.get_all_chunks()
        
        # Count files by language
        language_stats = {}
        for path, _ in chunks.items():
            if path.startswith('file:'):
                ext = path.split('.')[-1] if '.' in path else 'unknown'
                language_stats[ext] = language_stats.get(ext, 0) + 1
        
        # Display statistics
        table = Table(title="Repository Statistics")
        table.add_column("Language", style="cyan")
        table.add_column("Number of Files", style="magenta")
        
        for lang, count in sorted(language_stats.items()):
            table.add_row(lang, str(count))
            
        console.print(table)

async def query_repository(rag: CodeRAG, llm: GroqLLM, query: str):
    """Query repository and get LLM analysis."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        # Search for relevant code
        search_task = progress.add_task("[cyan]Searching codebase...", total=None)
        results = await rag.query_repository(
            query=query,
            max_results=5,
            similarity_threshold=0.7
        )
        progress.update(search_task, visible=False)
        
    if not results:
        console.print("[yellow]No relevant code found for your query.")
        return
        
    # Convert results to code contexts
    contexts = []
    for result in results:
        chunk = result['chunk']
        contexts.append(CodeContext(
            content=chunk['content'],
            language=chunk['language'],
            file_path=chunk['file_path']
        ))
    
    # Get LLM analysis
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        analysis_task = progress.add_task("[cyan]Analyzing code with LLM...", total=None)
        analysis = await llm.analyze_code(query, contexts)
        progress.update(analysis_task, visible=False)
    
    # Display results
    console.print("\n[bold cyan]Relevant Code Sections:[/bold cyan]")
    for ctx in contexts:
        console.print(Panel(
            f"[bold]File:[/bold] {ctx.file_path}\n[bold]Language:[/bold] {ctx.language}\n\n```{ctx.language}\n{ctx.content}\n```",
            title="Code Chunk",
            expand=False
        ))
        
    console.print("\n[bold cyan]LLM Analysis:[/bold cyan]")
    console.print(Panel(analysis, title="Analysis", expand=False))

async def interactive_mode(repo_url: str):
    """Interactive mode for continuous querying."""
    # Check environment variables
    required_vars = {
        "CLOUDFLARE_API_TOKEN": "Cloudflare API token",
        "CLOUDFLARE_EMAIL": "Cloudflare account email",
        "CLOUDFLARE_ACCOUNT_ID": "Cloudflare account ID",
        "OPENAI_API_KEY": "OpenAI API key for embeddings",
        "GROQ_API_KEY": "Groq API key"
    }
    
    missing_vars = []
    for var, desc in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{desc} ({var})")
            
    if missing_vars:
        console.print("[red]Missing required environment variables:")
        for var in missing_vars:
            console.print(f"[red]- {var}")
        return
    
    # Initialize RAG system with environment variables
    rag = CodeRAG(
        account_id=os.environ.get("CLOUDFLARE_ACCOUNT_ID"),
        openai_key=os.environ.get("OPENAI_API_KEY"),
        namespace="code-understanding"
    )
    
    llm = GroqLLM(api_key=os.environ.get("GROQ_API_KEY"))
    
    # Index the repository first
    await index_repository(rag, repo_url)
    
    # Interactive query loop
    console.print("\n[bold green]Repository indexed! You can now ask questions.[/bold green]")
    console.print("[cyan]Type 'exit' to quit.[/cyan]\n")
    
    while True:
        try:
            query = input("\n[?] Enter your question: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            if query.strip():
                await query_repository(rag, llm, query)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}")
    
    console.print("\n[green]Thank you for using Code Understanding AI![/green]")

async def main():
    parser = argparse.ArgumentParser(description="Code Repository Analysis Tool")
    parser.add_argument("--repo", type=str, help="GitHub repository URL to analyze")
    parser.add_argument("--query", type=str, help="Query to search in the codebase")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode with continuous querying")
    args = parser.parse_args()
    
    if args.interactive and args.repo:
        await interactive_mode(args.repo)
    elif args.repo and args.query:
        # Initialize and run single query
        rag = CodeRAG(
            account_id=os.environ.get("CLOUDFLARE_ACCOUNT_ID"),
            openai_key=os.environ.get("OPENAI_API_KEY"),
            namespace="code-understanding"
        )
        llm = GroqLLM(api_key=os.environ.get("GROQ_API_KEY"))
        
        await index_repository(rag, args.repo)
        await query_repository(rag, llm, args.query)
    else:
        console.print("""
[bold cyan]Code Repository Analysis Tool[/bold cyan]

This tool helps you analyze and understand code repositories using AI.

Usage:
1. Interactive mode (recommended):
   [green]python main.py --repo https://github.com/user/repo --interactive[/green]

2. Single query mode:
   [green]python main.py --repo https://github.com/user/repo --query "How does the authentication work?"[/green]

Make sure to set the following environment variables in .env:
- CLOUDFLARE_API_TOKEN
- CLOUDFLARE_EMAIL
- CLOUDFLARE_ACCOUNT_ID
- OPENAI_API_KEY
- GROQ_API_KEY
        """)

if __name__ == "__main__":
    asyncio.run(main())
