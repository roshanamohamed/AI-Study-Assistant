from fastapi import FastAPI
from pydantic import BaseModel
from utils.retriever import get_relevant_context
from langchain_google_vertexai import ChatVertexAI
import os

# ---- Vertex AI / Gemini config ----
# You can either:
# 1) Set GOOGLE_APPLICATION_CREDENTIALS env var to your service account json
# 2) Or rely on your local gcloud auth environment

# Optional: if you want to hardcode project/location, you can:
VERTEX_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "internship-knowledge-hub")
VERTEX_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Initialize Gemini via LangChain
model = ChatVertexAI(
    model="gemini-1.5-flash",
    project=VERTEX_PROJECT,
    location=VERTEX_LOCATION,
    temperature=0.2,
    max_output_tokens=400,
)

app = FastAPI(title="AI Study Assistant")


class Query(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "AI Study Assistant is running."}


@app.post("/ask")
def ask_question(query: Query):
    """
    Main endpoint: takes a question, fetches relevant context from notes,
    and asks Gemini to answer based on that context.
    """

    # 1. Get relevant notes
    context = get_relevant_context(query.question)

    # 2. Build prompt for the model
    prompt = f"""
You are a helpful AI Study Assistant. Use the context from the student's notes
to answer the question clearly and concisely. If the context does not contain
the answer, say you are not sure and give a high-level explanation instead.

CONTEXT FROM NOTES:
{context}

QUESTION:
{query.question}

ANSWER (student-friendly, step-by-step if needed):
"""

    # 3. Call Gemini through LangChain
    response = model.invoke(prompt)

    # response.content may be a string or list depending on version.
    # We handle the simple case where it's just text:
    answer = getattr(response, "content", str(response))

    return {
        "question": query.question,
        "context_used": context,
        "answer": answer,
    }
