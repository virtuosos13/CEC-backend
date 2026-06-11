from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from cec_vector_store import CECVectorStore
from cec_usage_tracker import CECUsageTracker

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
usage = CECUsageTracker(os.path.join(base_dir, "cec_users.db"))

class ChatRequest(BaseModel):
    message: str
    user_id: str  # Added for gatekeeping

class ChatResponse(BaseModel):
    answer: str
    citations: list
    status_message: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 1. Check Access (Gatekeeping)
    has_access, status = usage.check_access(request.user_id)
    if not has_access:
        return ChatResponse(
            answer="You have reached the limit of your trial version.",
            citations=[],
            status_message=status
        )

    # 2. Search PDF
    results = store.search(request.message, k=5)
    if not results:
        return ChatResponse(answer="No relevant rules found.", citations=[], status_message=status)

    # 3. Build Grounded Prompt
    context_text = "\n\n".join([
        f"[Rule {r['metadata']['rule']}, Page {r['metadata']['page']}]\n{r['text']}" 
        for r in results
    ])

    prompt = f"""
    You are an expert 2024 Canadian Electrical Code assistant.
    Answer the question strictly using the provided context.
    Cite the Rule and Page for every part of your answer.
    
    CONTEXT:
    {context_text}

    USER QUESTION:
    {request.message}
    """

    # 4. Generate & Track Usage
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Only charge/increment if we actually hit the AI
        usage.increment_usage(request.user_id)
        
        return ChatResponse(
            answer=response.text, 
            citations=[r['metadata'] for r in results],
            status_message=status
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
