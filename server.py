import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

app = Flask(__name__)
@app.route("/")
def home():
    return "Server running"
CORS(app)

# --- OpenRouter / Qwen Setup ---
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY not found in .env file.")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

FREE_MODELS = [
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-vl-plus:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "arcee-ai/trinity-large-preview:free",
    "deepseek/deepseek-r1-distill-llama-70b:free"    
]

# --- Local Embeddings & Chroma Vector Store ---
db_folder = "chroma_db"

if not os.path.exists(db_folder):
    raise RuntimeError(f"Chroma database not found at '{db_folder}'. Run ingest_local.py first.")

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(persist_directory=db_folder, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 9})
print("Backend ready. Listening on http://localhost:5000")


@app.route("/info", methods=["GET"])
def info():
    """Return chatbot metadata for the sidebar."""
    try:
        chunk_count = vectorstore._collection.count()
    except Exception:
        chunk_count = "N/A"
    return jsonify({
        "model": "Auto-switching (Free Models)",
        "chunks": chunk_count
    })


@app.route("/chat", methods=["POST"])
def chat():
    """
    Accepts: { "message": "...", "history": [{"role": "...", "content": "..."}] }
    Returns: { "reply": "..." }
    """
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Retrieve relevant chunks
    try:
        docs = retriever.invoke(user_message)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    if not docs:
        context = "No relevant information found."
    else:
        context = "\n\n".join([
            f"From {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}"
            for doc in docs
        ])

    system_prompt = f"""
You have the data about the Pagalavan. Who ever ask about him answer only from the data.
Answer ONLY using information from the provided context.
Rules:
- Be very brief and to the point, maximum 150 to 250 words.
- Never repeat the same fact twice in one answer.
- If information from different chunks relates, synthesize it naturally — do NOT list everything separately.
- Use at most 3 to 4 bullet points.
- If the question is clearly related to documented info, answer directly — do NOT say "I don't know" prematurely.
- If truly no relevant info exists → say: "I don't have information about that in my knowledge base."
- End with a short friendly note only if it adds value.

Relevant information:
{context}

Be friendly, concise and funny.
"""

    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    api_messages.append({"role": "user", "content": user_message})

    for model_name in FREE_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=api_messages
            )
            bot_reply = response.choices[0].message.content
            return jsonify({"reply": bot_reply})
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str:
                print(f"Model {model_name} rate limited. Trying next available...")
                continue
            else:
                print(f"Model {model_name} encountered an error: {e}")
                continue

    # If the loop finishes without returning, all models have failed or hit rate limits
    return jsonify({"error": "All available AI models are currently busy. Please try again in a few minutes."}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
