from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBED_MODEL, CUSTOM_INDEX_DIR
from huggingface_hub import InferenceClient

model_id = 'meta-llama/llama-3-3-70b-instruct' 


def retrieve_context(query, domain):
    db = FAISS.load_local(str(CUSTOM_INDEX_DIR), 
                          HuggingFaceEmbeddings(model_name=EMBED_MODEL),
                          allow_dangerous_deserialization=True     
    )
    
    results = db.similarity_search(query, k=3, filter={"domain": domain})
    return results

def generate_answer(context, question):
    
    client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token="hf_xxxxxxxxxxxxxxx"
    )
    for c in context:
        print("\n" + "="*80)
        print("File:", c.metadata["file_name"])
        print("Page:", c.metadata["page_1based"])

    response = client.chat_completion(
        messages=[
        {"role": "system", "content": "Απάντησε στα ελληνικά μόνο βάση του συγκεκριμένου content. "
        "Αν δεν βρεις την απάντηση πες ότι η απάντηση δεν είναι στο εύρος των γνώσεων σου"},
        {"role": "user",
         "content": f"Context:\n{context}\n\nΕρώτηση:\n{question}"}
    ],
    max_tokens=300,
    temperature=0.2
    )
    print(response.choices[0].message.content)
    for c in context:
        print("Πηγές:", c.metadata["file_name"], c.metadata["page_1based"])
