from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time


from bert_transformer import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INTENT_META = {
    "greeting": {
        "label": "Greeting",
        "emoji": "👋",
        "reply": "Hello! Welcome to support. How can I help you today?",
        "color": "#6366f1",
    },
    "password_reset": {
        "label": "Password Reset",
        "emoji": "🔐",
        "reply": "I can help you reset your password. Please visit our password reset page or check your email for a reset link.",
        "color": "#f59e0b",
    },
    "refund_request": {
        "label": "Refund Request",
        "emoji": "💸",
        "reply": "I've noted your refund request. Refunds are typically processed within 5–7 business days. A confirmation will be sent to your email.",
        "color": "#10b981",
    },
    "order_status": {
        "label": "Order Status",
        "emoji": "📦",
        "reply": "To track your order, please provide your order ID and I'll fetch the latest status for you.",
        "color": "#3b82f6",
    },
    "payment_issue": {
        "label": "Payment Issue",
        "emoji": "💳",
        "reply": "I'm sorry to hear about the payment issue. Please check your card details, ensure sufficient balance, and try again. If the issue persists, contact your bank.",
        "color": "#ef4444",
    },
    "complaint": {
        "label": "Complaint",
        "emoji": "😔",
        "reply": "I sincerely apologise for your experience. Your complaint has been logged and a senior agent will follow up with you within 24 hours.",
        "color": "#8b5cf6",
    },
}


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    intent: str
    intent_label: str
    intent_emoji: str
    intent_color: str
    reply: str
    confidence: float
    processing_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start = time.perf_counter()
    result = predict(req.message)
    processing_ms = int((time.perf_counter() - start) * 1000)

    meta = INTENT_META[result["intent"]]
    return ChatResponse(
        intent=result["intent"],
        intent_label=meta["label"],
        intent_emoji=meta["emoji"],
        intent_color=meta["color"],
        reply=meta["reply"],
        confidence=result["confidence"],
        processing_ms=processing_ms,
    )