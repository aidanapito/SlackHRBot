import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI client (new SDK)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up Chroma vector DB
chroma_client = chromadb.Client(Settings(persist_directory="chroma_store"))
collection = chroma_client.get_or_create_collection("hr_docs")

# Function to embed text using OpenAI
def embed_text(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding

# Load and chunk .txt files in hr_docs/
def load_docs():
    docs = []
    for filename in os.listdir("hr_docs"):
        if filename.endswith(".txt"):
            with open(f"hr_docs/{filename}", "r", encoding="utf-8") as f:
                docs.append((filename, f.read()))
    return docs

# Process and store
docs = load_docs()
for i, (filename, text) in enumerate(docs):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    for j, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": filename}],
            ids=[f"{filename}_{j}"],
            embeddings=[embed_text(chunk)]
        )

print("✅ HR documents embedded and stored in Chroma.")
