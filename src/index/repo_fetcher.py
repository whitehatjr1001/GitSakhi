import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

class RepoFetcher:
    def __init__(self, url, user_agent=None):
        self.url = self.convert_to_uithub(url)
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    @staticmethod
    def convert_to_uithub(url):
        parsed_url = urlparse(url)
        return url.replace("github.com", "uithub.com") if parsed_url.netloc == "github.com" else url

    def fetch_repo_content(self):
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(self.url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return str(soup)
        except requests.RequestException as e:
            return None

if __name__ == "__main__":
    url = "https://github.com/whitehatjr1001/ContentAI"
    fetcher = RepoFetcher(url)
    repo_content = fetcher.fetch_repo_content()
    
    if repo_content:
        print(f"Successfully fetched repository content")
        print(repo_content)
    else:
        print("Failed to fetch repository content")
