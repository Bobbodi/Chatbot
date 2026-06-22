from flask import Flask, render_template, request, jsonify
import chromadb
import ollama

app = Flask(__name__)

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_collection("knowledge")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json["message"]

    # Embed user question
    embedding = ollama.embed(
        model="nomic-embed-text",
        input=user_message
    )

    # Retrieve relevant documents
    results = collection.query(
        query_embeddings=[
            embedding["embeddings"][0]
        ],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
Use ONLY the provided context.

Context:
{context}

Question:
{user_message}

If the answer is not in the context, say:
"I don't know based on the provided knowledge."
"""

    response = ollama.chat(
        model="qwen2:0.5b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return jsonify({
        "response": response["message"]["content"]
    })


if __name__ == "__main__":
    app.run(debug=True)