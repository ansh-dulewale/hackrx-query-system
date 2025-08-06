import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import asyncio
from dotenv import load_dotenv

# Import our custom modules
from document_processor import process_pdf_from_url
from vector_manager import setup_pinecone, embed_and_store, query_vector_db, openai_client

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# --- Pydantic Models for Request and Response ---
class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

class AnswerResponse(BaseModel):
    answers: list[str]

# --- Helper function to get final answer from LLM ---
async def get_answer_from_llm(context: str, question: str) -> str:
    """
    Asks the LLM to answer a question based on the provided context.
    """
    print(f"Getting answer for question: {question}")
    try:
        completion = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in policy documents. Answer the user's question based *only* on the context provided. If the answer is not in the context, say 'The answer is not available in the provided context.'"},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error getting answer from LLM: {e}")
        return "Error processing the question."

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=AnswerResponse)
async def run_submission(request: QueryRequest, authorization: str = Header(None)):
    """
    The main endpoint to process a document and answer questions about it.
    """
    expected_token = f"Bearer {os.getenv('AUTH_TOKEN')}"
    if authorization != expected_token:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid authentication token")

    # 1. Initialize Pinecone
    pinecone_index = setup_pinecone()

    # 2. Process the document from the URL
    text_chunks = process_pdf_from_url(request.documents)
    if not text_chunks:
        raise HTTPException(status_code=400, detail="Could not process the document from the URL.")

    # 3. Embed and store the document chunks
    embed_and_store(text_chunks, pinecone_index)

    # 4. Process all questions concurrently
    answer_tasks = []
    for question in request.questions:
        # For each question, first find relevant context
        context = query_vector_db(question, pinecone_index)
        # Then, create a task to get the final answer from the LLM
        answer_tasks.append(get_answer_from_llm(context, question))
    
    answers = await asyncio.gather(*answer_tasks)

    return {"answers": answers}

# --- Root endpoint for simple testing ---
@app.get("/")
def read_root():
    return {"message": "HackRx Query System is running."}
