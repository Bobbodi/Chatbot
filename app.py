from flask import Flask, render_template, request, jsonify
import chromadb
import ollama
import time

app = Flask(__name__)

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_collection("knowledge")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json["message"]

    embed_start = time.time()
    # Embed user question
    embedding = ollama.embed(
        model="nomic-embed-text",
        input=user_message
    )
    embed_time = time.time() - embed_start

    retrieval_start = time.time()
    # Retrieve relevant documents
    results = collection.query(
        query_embeddings=[
            embedding["embeddings"][0]
        ],
        n_results=10
    )
    retrieval_time = time.time() - retrieval_start

    documents = results["documents"][0]
    distances = results["distances"][0]

    sources = []

    for doc, distance in zip(documents, distances):
        sources.append({
            "text": doc,
            "distance": distance
        })

    context = "\n".join(results["documents"][0])

    llm_start = time.time()
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": """
                You are a knowledge-base assistant.
                Answer using only the provided context. 
                Formatting: 
                - Be concise and direct.
                - Use numbers for lists. 
                - Use bold for key terms.
                If the answer is not in the context, do not make up information and say "I don't know".
                """
            },
            {
                "role": "user",
                "content": f"""
                Context:
                {context}

                Question:
                {user_message}
                """
            }
        ],
        options={
            "temperature": 0.1
        }
    )
    llm_time = time.time() - llm_start

    return jsonify({
        "response": response["message"]["content"],
        "sources": sources,
        "times": {
            "embed": round(embed_time, 3),
            "retrieval": round(retrieval_time, 3),
            "llm": round(llm_time, 3)
        }
    })


if __name__ == "__main__":
    app.run(debug=True)