import os
from pathlib import Path
from dotenv import load_dotenv
from rich import print

from src.rag.cloudflare_vectorize import CloudflareVectorize

def main():
    # Load environment variables from .env
    env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        print(f"[red]No .env file found at {env_path}")
        return
        
    load_dotenv(env_path)
    
    # Required environment variables
    required_vars = {
        "CLOUDFLARE_API_KEY": "your Cloudflare API key (Global API Key)",
        "CLOUDFLARE_EMAIL": "your Cloudflare account email", 
        "CLOUDFLARE_ACCOUNT_ID": "your Cloudflare account ID"
    }
    
    # Check for missing variables
    missing = []
    for var, desc in required_vars.items():
        if not os.getenv(var):
            missing.append(f"{var} ({desc})")
            
    if missing:
        print("[red]Missing required environment variables:")
        for var in missing:
            print(f"- {var}")
        return
        
    try:
        # Initialize Cloudflare client
        vectorize = CloudflareVectorize(index_name="example-index")
        
        # Delete index
        print("\n[yellow]Deleting vector index...")
        result = vectorize.delete_index()
        print(f"[green]{result['message']}")
        
    except Exception as e:
        print(f"[red]Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
