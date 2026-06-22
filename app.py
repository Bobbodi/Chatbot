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

    documents = results["documents"][0]
    distances = results["distances"][0]

    sources = []

    for doc, distance in zip(documents, distances):
        sources.append({
            "text": doc,
            "distance": distance
        })

    context = "\n".join(results["documents"][0])

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
            - Use bullet points for lists. 
            - Use bold for key terms.
            If the answer is not in the context, say "I don't know".
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

    return jsonify({
        "response": response["message"]["content"],
        "sources": sources
    })


if __name__ == "__main__":
    app.run(debug=True)