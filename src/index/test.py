from repo_fetcher import RepoFetcher
from indexer import Indexer

def main():
    # Fetch repository content
    url = "https://github.com/whitehatjr1001/ContentAI"
    fetcher = RepoFetcher(url)
    repo_content = fetcher.fetch_repo_content()
    
    if not repo_content:
        print("Failed to fetch repository content")
        return
        
    # Create and initialize indexer
    indexer = Indexer()
    indexer.parse_content(repo_content)
    
    # Print repository structure
    print("\nRepository Structure:")
    print("-" * 50)
    print(indexer.get_structure_text())
    
    # Get all chunks
    chunks = indexer.get_all_chunks()
    
    # Print some example file contents
    print("\nExample File Contents:")
    print("-" * 50)
    
    # Show content of a few important files
    for item in indexer.repo_structure:
        if item['type'] == 'file':
            file_path = item['path']
            content = indexer.get_file_content(file_path)
            if content:
                print(f"\n{file_path}:")
                print("-" * len(file_path))
                print(content)

if __name__ == "__main__":
    main()
