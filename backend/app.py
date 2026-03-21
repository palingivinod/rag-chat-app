from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from rag.agents import rag_agent
from rag.retriever import add_documents
from langchain_community.document_loaders import PyPDFLoader , WebBaseLoader
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(query: str):
    return rag_agent(query)

@app.post("/upload")
async def upload(file: UploadFile):
    print("File received:", file.filename)
    path = f"data/docs/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(docs)
    print("📄 Loaded docs:", len(docs))
    # ✅ Add metadata for multi-doc tracking
    for doc in docs:
        doc.metadata["source"] = file.filename

    add_documents(docs)

    return {"message": f"{file.filename} uploaded successfully"}


@app.post("/upload-url")
async def upload_url(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()

    print(" Loaded URL docs:", len(docs))


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    # Add metadata
    for doc in split_docs:
        doc.metadata["source"] = url

    add_documents(split_docs)

    return {"message": "URL content added successfully"}



class EmailRequest(BaseModel):
    email: str
    content: str


@app.post("/send-email")
def send_email(data: EmailRequest):
    sender = ""
    password = ""

    msg = MIMEText(data.content)
    msg["Subject"] = "RAG Assistant Response"
    msg["From"] = sender
    msg["To"] = data.email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.send_message(msg)

    return {"message": "Email sent"}

@app.post("/reset")
def reset():
    global vectorstore
    from rag.retriever import vectorstore
    vectorstore = None
    return {"message": "Reset successful"}
