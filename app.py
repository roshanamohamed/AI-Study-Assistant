import io
import os
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pypdf import PdfReader

from langchain_google_vertexai import ChatVertexAI

from utils.retriever import init_vectorstore, get_relevant_context, add_text_document

# ---- Vertex AI / Gemini config ----
VERTEX_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "internship-knowledge-hub")
VERTEX_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

model = ChatVertexAI(
    model="gemini-1.5-flash",
    project=VERTEX_PROJECT,
    location=VERTEX_LOCATION,
    temperature=0.2,
    max_output_tokens=500,
)

app = FastAPI(title="AI Study Assistant")

# Static + templates setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class Query(BaseModel):
    question: str


@app.on_event("startup")
def startup_event():
    # Build initial vector store from data/notes.txt etc.
    init_vectorstore()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "AI Study Assistant"},
    )


@app.post("/ask")
async def ask_question(query: Query):
    """
    Takes a question, retrieves relevant context from the vector store,
    and uses Gemini to generate an answer.
    """
    context = get_relevant_context(query.question)

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

    response = model.invoke(prompt)
    answer = getattr(response, "content", str(response))

    return {
        "answer": answer,
        "context": context,
    }


def _extract_text_from_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)


@app.post("/upload")
async def upload_notes(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """
    Accepts either:
      - a PDF file (lecture slides / notes)
      - raw text from a textarea
    and adds it into the vector store.
    """
    combined_text = ""

    # If file uploaded
    if file is not None:
        raw = await file.read()
        if file.content_type == "application/pdf":
            pdf_text = _extract_text_from_pdf_bytes(raw)
            combined_text += "\n" + pdf_text
        else:
            try:
                combined_text += "\n" + raw.decode("utf-8", errors="ignore")
            except Exception:
                return JSONResponse(
                    {"status": "error", "message": "Unsupported file encoding."},
                    status_code=400,
                )

    # If raw text provided
    if text:
        combined_text += "\n" + text

    if not combined_text.strip():
        return JSONResponse(
            {"status": "error", "message": "No content provided."},
            status_code=400,
        )

    add_text_document(combined_text)

    return {"status": "ok", "message": "Notes added to your study assistant!"}
