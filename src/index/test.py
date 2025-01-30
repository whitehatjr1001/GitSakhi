from shlex import join
from bs4 import BeautifulSoup
import requests

url = "https://uithub.com/whitehatjr1001/ContentAI"
response = requests.get(url)
html_content = response.text
repo = ''
soup = BeautifulSoup(html_content, 'html.parser')
repo = ''.join(str(soup))
print(repo)
print(len(repo))