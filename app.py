import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client (v1+ syntax)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Slack app
app = App(token=os.getenv("SLACK_BOT_TOKEN"))

# Load Chroma vector DB
chroma_client = chromadb.Client(Settings(persist_directory="chroma_store"))
collection = chroma_client.get_or_create_collection("hr_docs")

# Embed a query using OpenAI
def embed_query(query):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return response.data[0].embedding

# Generate an answer using GPT-4
def get_answer(query):
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4,
    )
    docs = results["documents"][0]
    context = "\n\n".join(docs)

    prompt = f"""
You are an HR assistant. Use the following HR policy context to answer the employee's question. Be concise and clear.

Context:
{context}

Question: {query}
Answer:
"""

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content

# Handle Slack mentions
@app.event("app_mention")
def handle_message_events(body, say):
    print("📩 EVENT RECEIVED:", body)

    user_query = body["event"]["text"]
    user_query = user_query.replace(f"<@{body['event']['user']}>", "").strip()

    say("🤖 Thinking...")
    try:
        answer = get_answer(user_query)
        say(answer)
    except Exception as e:
        say(f"⚠️ Error: {str(e)}")

# Run the Slack bot
if __name__ == "__main__":
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()
