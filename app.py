import os
import json
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from rapidfuzz import process

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Slack app
app = App(token=os.getenv("SLACK_BOT_TOKEN"))

# Load Chroma vector DB
chroma_client = chromadb.Client(Settings(persist_directory="chroma_store"))
collection = chroma_client.get_or_create_collection("hr_docs")

# Load and flatten section_map.json
alias_to_file = {}
try:
    with open("config/section_map.json", encoding="utf-8") as f:
        sm = json.load(f)
    for section in sm.get("sections", []):
        filename = section["filename"]
        for alias in section.get("aliases", []):
            alias_to_file[alias.lower()] = filename
except FileNotFoundError:
    print("⚠️ config/section_map.json not found. Exact section matching will be disabled.")

def get_best_matching_section(query, threshold=70):
    if not alias_to_file:
        return None
    best = process.extractOne(query.lower(), alias_to_file.keys())
    if best and best[1] >= threshold:
        return alias_to_file[best[0]]
    return None

# Embed a query using OpenAI
def embed_query(query):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query  # string per new SDK
    )
    return response.data[0].embedding

def get_answer(query):
    matched_section = get_best_matching_section(query)

    if matched_section:
        section_path = os.path.join("hr_docs_by_section", matched_section)
        if not os.path.exists(section_path):
            return f"⚠️ Section file not found: {matched_section}"

        with open(section_path, "r", encoding="utf-8") as f:
            context = f.read()

        prompt = f"""
You are an HR assistant. Use only the following HR policy section to answer the employee's question. Do not guess; if the answer is not present, say so.

Context:
{context}

Question:
{query}

Answer:
"""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    # Fallback to vector search
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4,
    )
    docs = results["documents"][0]
    if not docs:
        return "I couldn't find relevant policy information for that question."

    context = "\n\n".join(docs)
    prompt = f"""
You are an HR assistant. Use the following HR policy content to answer the employee's question. Do not hallucinate.

Context:
{context}

Question:
{query}

Answer:
"""
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# Handle Slack mentions
@app.event("app_mention")
def handle_message_events(body, say):
    print("EVENT RECEIVED:", body)

    user_query = body["event"]["text"]
    user_query = user_query.replace(f"<@{body['event']['user']}>", "").strip()

    say("🤖 Thinking...")
    try:
        answer = get_answer(user_query)
        say(answer)
    except Exception as e:
        say(f"Error: {str(e)}")

# Run the Slack bot
if __name__ == "__main__":
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()
