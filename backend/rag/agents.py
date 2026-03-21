import os
from groq import Groq
from rag.retriever import retrieve
from rag.memory import get_memory, add_to_memory

apikey = ""
client = Groq(api_key=apikey)

def has_uploaded_documents():
    
    try:
        
        collection = ""  # Replace with your actual collection
        count = collection.count()
        return count > 0
    except:
        # If using simple file system check
        import os
        docs_folder = "data/docs"  # Replace with your actual folder
        if os.path.exists(docs_folder):
            return len(os.listdir(docs_folder)) > 0
        return False

def generate_llm(prompt):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def rag_agent(query):
    # Check if documents are uploaded
    docs_exist = has_uploaded_documents()
    
    # If no documents uploaded, skip RAG entirely
    if not docs_exist:
        prompt = f"""
You are a helpful AI assistant.

give best and good answers

Conversation History:
{get_memory_history()}

Question: {query}

INSTRUCTIONS:
- use small tables  for comparison or details
- No lengthy explanations
- Just answer directly
- Be conversational and friendly
"""
        answer = generate_llm(prompt)
        final = {
            "answer": answer,
            "confidence": 100,
            "source": ["General Knowledge"]
        }
        add_to_memory(query, final["answer"])
        return final
    
    # If documents exist, proceed with normal RAG
    docs, score = retrieve(query)

    context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
    sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))

    memory = get_memory()
    history = "\n".join([f"User: {m['user']}\nBot: {m['bot']}" for m in memory])

    query_lower = query.lower()

    # Detect section-based query
    is_section_query = any(word in query_lower for word in [
        "introduction", "conclusion", "summary", "abstract", "section"
    ])

    # Detect if user wants tables or detailed info
    wants_details = any(word in query_lower for word in [
        "table", "comparison", "compare", "list all", "detailed", "explain in detail"
    ])

    if score >= 0.5:
        if wants_details:
            prompt = f"""
You are an intelligent document assistant.

Use ONLY the provided context to answer the question.

Conversation History:
{history}

Context:
{context}

Question:
{query}

INSTRUCTIONS:
- Provide detailed information with tables if appropriate
- Be structured but clear
- Do NOT hallucinate
"""
        else:
            prompt = f"""
You are an intelligent document assistant.

Use ONLY the provided context to answer the question.

Conversation History:
{history}

Context:
{context}

Question:
{query}

INSTRUCTIONS:
- give complete and good answers from given source
- Avoid tables unless absolutely necessary
- Be concise and direct
- Do NOT hallucinate
- Just answer the question simply
"""

        answer = generate_llm(prompt)

        final = {
            "answer": answer,
            "confidence": int(score * 100),
            "source": sources
        }

    elif score >= 0.2:
        if wants_details:
            prompt = f"""
You are an AI assistant.

Use the context below to answer the question.
Add general knowledge to improve explanation.

Conversation History:
{history}

Context:
{context}

Question:
{query}

INSTRUCTIONS:
- Provide helpful information
- Use tables if it helps organize data
- Be clear and structured
"""
        else:
            prompt = f"""
You are an AI assistant.

Use the context below to answer the question.

Conversation History:
{history}

Context:
{context}

Question:
{query}

INSTRUCTIONS:
- give good answers based on the context
- use exact content form the document 
- Avoid tables if not necessary
- Be concise and direct
- Add "(Note: Based on partial document match)" at the end
"""

        answer = generate_llm(prompt)

        final = {
            "answer": answer,
            "confidence": int(score * 100),
            "source": sources if sources else ["Partial Match"]
        }

    else:
        # Low confidence - answer from general knowledge
        if wants_details:
            fallback_prompt = f"""
Answer this question using general knowledge.
Provide helpful information.

Question: {query}

INSTRUCTIONS:
- Give a helpful answer
- Use tables if appropriate
- Be informative
"""
        else:
            fallback_prompt = f"""
Answer this question using general knowledge.

Question: {query}

INSTRUCTIONS:
- Give a good answer based on your knowledge
- NO tables unless specifically asked
- NO lengthy explanations
- Just answer directly
- Be conversational
"""

        answer = generate_llm(fallback_prompt)

        final = {
            "answer": answer,
            "confidence": int(score * 100),
            "source": ["General Knowledge"]
        }

    # Save memory
    add_to_memory(query, final["answer"])

    return final

def get_memory_history():
    memory = get_memory()
    return "\n".join([f"User: {m['user']}\nBot: {m['bot']}" for m in memory[-5:]])  # Last 5 conversations