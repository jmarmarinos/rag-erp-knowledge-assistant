from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBED_MODEL, CUSTOM_INDEX_DIR, USER_GUIDES_DIR
from huggingface_hub import InferenceClient
from vectorstores import load_pdfs_with_metadata
import re

model_id = 'meta-llama/llama-3-3-70b-instruct' 

def detect_domain(question):
#function to detect the domain for semantic retrieval based on key words
    q = question.lower()

    mrp_terms = ['mrp', 'παραγγελία αγοράς', 'sales order', 'lead time']
    mps_terms = ['mps', 'προγραμματισμός παραγωγής', 'bom', 'routing', 'γραμμή παραγωγής']
    wms_terms = ['wms','sscc', 'lot', 'αποθήκη', 'ζώνη','picking','bin','παραλαβή','παραλαβή αποθήκης','παραλαμβάνω']

    if any(m in q for m in mrp_terms): 
        domain = 'mrp'
    elif any(m in q for m in mps_terms): 
        domain = 'mps'
    elif any(m in q for m in wms_terms): 
        domain = 'wms'
    else:
        domain = 'uknown'

    return domain

def detect_doc_type(question):
#function to detect document type (blueprint vs userguides)
    q = question.lower().strip()

    userguide_pattern = [
        r'\bπως να\b', 
        r'\bβοήθα με\b', 
        r'\b ποια βήματα\b',
        r'\bπως μπορώ\b',
        r'\bπως παραλαμβάνω\b',
        r'\bπως καταχωρώ\b',
        r'\bπως γίνεται\b'
        ]
    blueprint_pattern = [
        r'\bτι είναι\b', 
        r'\bπως υπολογίζεται\b', 
        r'\bτι μπορεί\b',
        r'\bνα φταίει\b',
        r'\bποια είναι\b'
        ]
    
    blueprint_hits = [p for p in blueprint_pattern if re.search(p,q)]
    userguide_hits = [p for p in userguide_pattern if re.search(p,q)]

    if len(blueprint_hits) > len(userguide_hits):
        return{
            "doc_type" : 'blueprint',
            "confidence" : len(blueprint_hits),
            "matched_patterns" : blueprint_hits

        }
    else:
        return{
            "doc_type" : "userguide",
            "confidence" : len(userguide_hits),
            "matched_patterns" : userguide_hits
        }
def expand_whole_doc(results):

    whole_doc = load_pdfs_with_metadata(USER_GUIDES_DIR, "userguide")

    whole_doc = sorted(
        whole_doc,
        key=lambda d: (d.metadata["file_name"], d.metadata["page_1based"])
    )
    return whole_doc

def expand_doc_retrieval(results):
# Expand doc retrieval in case of user guides to include neighbor pages
    expanded_docs = []
    total_pages = set()
    all_docs =  load_pdfs_with_metadata(USER_GUIDES_DIR, "userguide")
    # Step 1: collect retrieved pages + neighbors
    for r in results:
        file_name = r.metadata["file_name"]
        page = r.metadata["page_1based"]

        for p in range(page - 1, page + 2):
            total_pages.add(file_name, p)
            
    # Step 2: keep matching docs from all_docs
    for a in all_docs:
        file_name = a.metadata["file_name"]
        page = a.metadata["page_1based"]
        if page in total_pages:
            expanded_docs.append(a)

    # Step 3: sort by file and page
    expanded_docs = sorted(
        expanded_docs,
        key=lambda d: (d.metadata["file_name"], d.metadata["page_1based"])
    )
    for e in expanded_docs:
        print("\n" + "="*80)
        print("File:", e.metadata["file_name"])
        print("Page:", e.metadata["page_1based"])
    return expanded_docs      

def retrieve_context(query, domain=None, doc_type=None):
    db = FAISS.load_local(str(CUSTOM_INDEX_DIR), 
                          HuggingFaceEmbeddings(model_name=EMBED_MODEL),
                          allow_dangerous_deserialization=True     
    )
    filter_dict = {}
    if doc_type is not None:
        filter_dict['doc_type'] = doc_type["doc_type"]
    
    if domain is not None:
        filter_dict['domain'] = domain

    print('Domain: ', domain)


    #get more results for userguides so no step is omitted
    if doc_type["doc_type"] == 'userguide':
        top_k = 15
    else: 
        top_k = 8

    if filter_dict:
        results = db.similarity_search(query, k=top_k, filter=filter_dict)
    else:
        results = db.similarity_search(query, k=top_k)

    #Sort the retrieved chunks based on their page number for userguides
    if doc_type["doc_type"] == 'userguide':
        results = sorted(results, key=lambda d: d.metadata["page_1based"])

    return results, doc_type["doc_type"]

def generate_answer(context, question):
    
    client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    )
    '''for c in context:
        print("\n" + "="*80)
        print("File:", c.metadata["file_name"])
        print("Page:", c.metadata["page_1based"])'''

    response = client.chat_completion(
        messages=[
        {"role": "system", 
         "content": '''Απάντησε στα ελληνικά μόνο με βάση το context.
                    Δομή απάντησης:
                    - Σύντομη απάντηση
                    - Βήματα (Αν η ερώτηση αφορά οδηγίες χρήστη, χρησιμοποίησε μόνο τα διαθέσιμα αποσπάσματα του οδηγού. Παρουσίασε τα βήματα με τη σωστή σειρά, με βάση τη σειρά των σελίδων. Για κάθε βήμα, ανέφερε τα πεδία που πρέπει να συμπληρώσει ο χρήστης μόνο εφόσον αναφέρονται ρητά στο κείμενο.)
                    - Παράδειγμα μόνο αν υπάρχει (εφόσον η ερώτηση αφορά οδηγίες χρήστη μην αναφέρεις παράδειγμα)
                    Αν δεν υπάρχει πληροφορία, πες το καθαρά.
                    Αν η ερώτηση περιγράφει ένα πρόβλημα ή λανθασμένο
                      αποτέλεσμα και το περιεχόμενο περιλαμβάνει συνθήκες ρύθμισης
                       ή παραμέτρους, απάντησε παραθέτοντας τις παραμέτρους,
                         τις προϋποθέσεις ή τους παράγοντες που επηρεάζουν και 
                         αναφέρονται στα έγγραφα, αντί να επινοείς μια
                           συγκεκριμένη διάγνωση.


                    '''},
        {"role": "user",
         "content": f"Context:\n{context}\n\nΕρώτηση:\n{question}"}
    ],
    max_tokens=500,
    temperature=0.2
    )
    print(response.choices[0].message.content)
    #for c in context:
    #    print("Πηγές:", c.metadata["file_name"], c.metadata["page_1based"])
