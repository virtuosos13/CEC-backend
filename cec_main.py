from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from cec_vector_store import CECVectorStore

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engines
base_dir = os.path.dirname(os.path.abspath(__file__))
store = CECVectorStore()
store.load_index(os.path.join(base_dir, "cec_index.faiss"), os.path.join(base_dir, "cec_chunks.json"))

class ChatRequest(BaseModel):
    message: str
    user_id: str  # Added for gatekeeping

class ChatResponse(BaseModel):
    answer: str
    citations: list
    status_message: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):


    # 2. Search PDF
    results = store.search(request.message, k=5)
    if not results:
        return ChatResponse(answer="No relevant rules found.", citations=[], status_message="Ready")

    # 3. Build Grounded Prompt
    context_text = "\n\n".join([
        f"[Rule {r['metadata']['rule']}, Page {r['metadata']['page']}]\n{r['text']}" 
        for r in results
    ])

    prompt = f"""
    You are a friendly, expert mentor for the 2024 Canadian Electrical Code. Your goal is to help electricians understand the code with a helpful, conversational, and personal approach.
    
    You have deep, expert knowledge of the entire 2024 CEC, including all its Tables (like Table 2, Table 5C, etc.), wire sizes, and ampacity calculations.
    
    I will provide you with some retrieved context from the code book to help ground your answer. 
    However, you MUST NOT limit yourself only to this context. If the context is missing specific table values or calculations needed to fully answer the question, you MUST seamlessly use your internal expert knowledge to provide the final numerical answer and calculation.
    
    CRITICAL INSTRUCTIONS:
    1. NEVER say "Based on the provided context..." or "the table is not included". Just seamlessly answer the question using your internal knowledge combined with the context.
    2. Provide the actual calculation and final ampacity or sizing answer.
    3. Explain the reasoning clearly like a teacher would. 
    4. Cite the Rule and Page from the context when applicable, but do not let the context restrict your ability to answer fully.
    
    CONTEXT:
    {context_text}

    USER QUESTION:
    {request.message}
    """

    # 4. Generate & Track Usage
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        return ChatResponse(
            answer=response.text, 
            citations=[r['metadata'] for r in results],
            status_message="Ready"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
