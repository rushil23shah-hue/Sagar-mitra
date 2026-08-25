"""
Sagar Mitra Backend - FastAPI Entrypoint
==========================================
Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then your frontend teammate hits:
    POST http://localhost:8000/api/chat   {"message": "..."}
    GET  http://localhost:8000/api/health
"""
import os
import traceback

from dotenv import load_dotenv
load_dotenv()  # reads .env in the working directory, if present

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.data import store
from app.chat import sagar_mitra_router

app = FastAPI(title="Sagar Mitra API", version="0.1.0")

# --- CORS ---------------------------------------------------------------
# Wide open for the hackathon so the frontend team (running on a different
# origin/port, e.g. localhost:5173 or a deployed static site) can call this
# freely. Tighten `allow_origins` to your actual frontend URL before any
# public/production use.
ALLOWED_ORIGINS = os.environ.get("SAGAR_MITRA_ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static files (trend-graph PNGs from Phase 3) ------------------------
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    mode: str  # "live_conditions" | "fallback_general_knowledge" | "concept_explanation" | "historical" | "comparison"
    query_context: dict | None = None
    statistical_result: dict | None = None
    plot_url: str | None = None
    confidence: str | None = None       # "HIGH" | "MODERATE" | "LOW" -- for the confidence badge
    alert: dict | None = None            # {"level": "SAFE"|"CAUTION"|"WARNING", "message": str}
    map_data: dict | None = None         # {"query_location": {lat,lon}, "nearby_points": [{lat,lon,temp,category}]}


@app.get("/")
def root():
    return {"service": "Sagar Mitra API", "status": "running"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gemini_key_configured": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS")),
        "using_synthetic_data": store.using_synthetic_data,
        "final_dataset_rows": len(store.final_dataset),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="`message` must not be empty.")

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS")):
        raise HTTPException(
            status_code=503,
            detail="No Gemini API key configured. Set GEMINI_API_KEY or GEMINI_API_KEYS in .env.",
        )

    try:
        result = sagar_mitra_router(req.message)
    except Exception as e:
        # Surface a clean 500 instead of letting the frontend see a raw
        # traceback / hang -- but log the full trace server-side for you.
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return ChatResponse(
        reply=result.get("reply", ""),
        mode=result.get("mode", "unknown"),
        query_context=result.get("query_context"),
        statistical_result=result.get("statistical_result"),
        plot_url=result.get("plot_url"),
        confidence=result.get("confidence"),
        alert=result.get("alert"),
        map_data=result.get("map_data"),
    )
