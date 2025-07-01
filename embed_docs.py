import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import tiktoken

load_dotenv()

#Tokenizer setup
encoding = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 1600

#OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#Chroma setup
chroma_client = chromadb.Client(Settings(persist_directory="chroma_store"))
collection = chroma_client.get_or_create_collection("hr_docs")

#Embedding function
def embed_text(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

#Load documents from folder
def load_docs():
    docs = []
    for filename in os.listdir("hr_docs_by_section"):
        if filename.endswith(".txt"):
            with open(os.path.join("hr_docs_by_section", filename), "r", encoding="utf-8") as f:
                docs.append((filename, f.read()))
    return docs

#Chunk text by token-safe sentences
def chunk_text(text, max_tokens=MAX_TOKENS):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        candidate = current_chunk + sentence + ". "
        if len(encoding.encode(candidate)) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

#Process & store in Chroma
docs = load_docs()
for i, (filename, text) in enumerate(docs):
    chunks = chunk_text(text)
    for j, chunk in enumerate(chunks):
        titled_chunk = f"Section: {filename.replace('_', ' ').title()}\n\n{chunk}"
        safe_id = f"{os.path.splitext(filename)[0].lower()}_{j}"
        collection.add(
            documents=[titled_chunk],
            metadatas=[{"source": filename}],
            ids=[safe_id],
            embeddings=[embed_text(chunk)]
        )

print("HR documents embedded and stored in Chroma.")
