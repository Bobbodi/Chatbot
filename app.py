from flask import Flask, render_template, request, jsonify
import chromadb
import ollama
import time
from sentence_transformers import CrossEncoder

app = Flask(__name__)

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_collection("knowledge")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

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
        query_embeddings=[embedding["embeddings"][0]],
        n_results=10
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    pairs = [
        (user_message, doc)
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(documents, distances, scores),
        key=lambda x: x[2],
        reverse=True
    )

    top_results = reranked[:3]

    context = "\n".join(
        doc
        for doc, _, _ in top_results
    )

    sources = [
        {
            "text": doc,
            "distance": float(distance),
            "rerank_score": float(score)
        }
        for doc, distance, score in top_results
    ]
    retrieval_time = time.time() - retrieval_start

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