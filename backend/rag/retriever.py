
from langchain_community.vectorstores import FAISS
from rag.embeddings import get_embeddings

vectorstore = None

def init_vectorstore(docs):
    global vectorstore
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

def add_documents(docs):
    global vectorstore

    if vectorstore:
        vectorstore.add_documents(docs)
    else:
        init_vectorstore(docs)

def retrieve(query):
    global vectorstore

    if not vectorstore:
        return [], 0

    results = vectorstore.similarity_search_with_score(query, k=3)

    docs = [r[0] for r in results]
    scores = [r[1] for r in results]

    similarity = 1 / (1 + min(scores))

    return docs, similarity
