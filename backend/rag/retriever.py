# from langchain_community.vectorstores import FAISS

# from rag.embeddings import get_embeddings

# import os
# VECTOR_PATH = "backend/faiss_index"
# vectorstore = None

# def init_vectorstore(docs):
#     global vectorstore
#     embeddings = get_embeddings()
#     vectorstore = FAISS.from_documents(docs, embeddings)

# def load_vectorstore():
#     global vectorstore
#     embeddings = get_embeddings()

#     if os.path.exists(VECTOR_PATH):
#         vectorstore = FAISS.load_local(
#             VECTOR_PATH,
#             embeddings,
#             allow_dangerous_deserialization=True
#         )

# def add_documents(docs):
#     global vectorstore

#     if vectorstore is None:
#         load_vectorstore()

#     if vectorstore:
#         vectorstore.add_documents(docs)
#     else:
#         init_vectorstore(docs)

#     vectorstore.save_local(VECTOR_PATH)

# def retrieve(query):
#     global vectorstore

#     if vectorstore is None:
#         load_vectorstore()

#     if not vectorstore:
#         print(" No vectorstore")
#         return [], 0

#     results = vectorstore.similarity_search_with_score(query, k=3)

#     if not results:
#         return [], 0

#     docs = [r[0] for r in results]
#     scores = [r[1] for r in results]

#     print("🔍 Raw scores:", scores)

#     # ✅ FIX: Convert FAISS distance → similarity properly
#     best_score = min(scores)   # smaller = better

#     similarity = 1 / (1 + best_score)

#     print("📊 Similarity:", similarity)

#     return docs, similarity

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