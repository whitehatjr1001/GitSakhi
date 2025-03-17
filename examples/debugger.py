import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Required API Credentials
CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
CLOUDFLARE_EMAIL = os.getenv("CLOUDFLARE_EMAIL")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")

# Check if credentials are missing
missing_vars = [var for var in ["CLOUDFLARE_API_KEY", "CLOUDFLARE_EMAIL", "CLOUDFLARE_ACCOUNT_ID"]
                if not os.getenv(var)]

if missing_vars:
    print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
    exit(1)

# Define the Cloudflare Vectorize API URL
VECTORIZE_URL = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/vectorize/namespaces"

# Create the namespace payload
payload = {
    "name": "my-debug-namespace",
    "vector_size": 1536,
    "metric": "cosine"
}

# Define request headers
headers = {
    "X-Auth-Email": CLOUDFLARE_EMAIL,
    "X-Auth-Key": CLOUDFLARE_API_KEY,
    "Content-Type": "application/json"
}

try:
    print("🚀 Sending request to create Cloudflare Vectorize namespace...\n")

    # Send the POST request
    response = requests.post(VECTORIZE_URL, headers=headers, json=payload)

    # Debugging logs
    print(f"🔍 Request URL: {VECTORIZE_URL}")
    print(f"📜 Request Headers: {headers}")
    print(f"📦 Request Payload: {json.dumps(payload, indent=2)}")

    # Check for errors
    if response.status_code == 200:
        print("\n✅ Namespace created successfully!")
        print(f"📄 Response: {json.dumps(response.json(), indent=2)}")
    else:
        print("\n❌ Failed to create namespace.")
        print(f"🔴 HTTP Status: {response.status_code}")
        print(f"⚠️ Response: {response.text}")

except requests.RequestException as e:
    print(f"\n🚨 Request failed due to an error: {e}")
