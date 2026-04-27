from flask import Flask, request, jsonify
import gradio as gr
from rag_pipeline import (
    retrieve_context, 
    generate_answer, 
    detect_domain, 
    detect_doc_type, 
    expand_doc_retrieval,
    expand_whole_doc
)

app = Flask(__name__)

@app.route("/ask")
def ask_questions(query: str) -> str:

    #This is added to support Flask API in the future
    #query = request.args.get("query")

    if not query:
        return ("Error: Question is required")
        #jsonify({"Error: Question is required"}), 400

    domain = detect_domain(query)

    doc_type = detect_doc_type(query)

    results = retrieve_context(query, domain, doc_type)

    if doc_type["doc_type"] == "userguide":
        results = expand_doc_retrieval(results)
        #results = expand_whole_doc(results)
    answer = generate_answer(results, query)

    return(answer)
    #return jsonify({
    #    "query": query,
    #    "answer": answer
    #})

def ui():
    iface = gr.Interface(
        fn = ask_questions,
        inputs=["text"],
        outputs = ["text"]
    )
    iface.launch(server_name = "127.0.0.1", server_port = 7860)

if __name__ == "__main__":
    ui()
