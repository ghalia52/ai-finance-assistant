import os
import re
import json
import uuid
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine, text, Column, String
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from contextlib import contextmanager, asynccontextmanager

# =========================
# CONFIGURATION
# =========================
class Settings(BaseSettings):
    # Gemini LLM API - REQUIRED, no defaults for security
    gemini_api_key: str
    gemini_enabled: bool = True
    gemini_api_url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    gemini_model: str = "gemini-1.5-pro"

    # Database - REQUIRED password, no default
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "finance_db"
    db_user: str = "postgres"
    db_password: str  # NO DEFAULT - must be set in .env
    db_driver: str = "postgresql+psycopg2"
    db_echo: bool = False

    # Database Table Configuration
    expenses_table_name: str = "transactions"
    transaction_id_column: str = "Transaction ID"
    transaction_date_column: str = "Date"
    transaction_currency_column: str = "Currency"
    transaction_sender_column: str = "Sender"
    transaction_receiver_column: str = "Receiver"
    transaction_amount_column: str = "Amount"
    transaction_fee_column: str = "Fee"
    transaction_type_column: str = "Type"

    # Finance defaults
    default_starting_balance: float = 1200.0
    default_category: str = "misc"
    currency_symbol: str = "$"
    currency_symbols: List[str] = ["$", "€", "£", "¥", "₹"]
    budget_warning_threshold: float = 0.8

    # LLM
    llm_max_tokens: int = 1000
    llm_temperature: float = 0.7
    llm_top_k: int = 40
    llm_top_p: float = 0.9
    llm_enabled: bool = True

    # App
    app_title: str = "AI Financial Assistant with Database"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: List[str] = ["*"]
    log_level: str = "INFO"

    # Chat
    max_chat_history: int = 24
    recent_chat_context: int = 6
    recent_expenses_limit: int = 5
    max_expenses_display: int = 10
    default_user_id: str = "anon"

    # System prompts
    system_prompt: str = "You are Finley, a friendly personal finance assistant. You can log expenses, set budgets, summarize spending, and answer general finance questions clearly and concisely. Always be practical and avoid giving legal or investment advice."

    llm_unavailable_message: str = "I'm sorry, the AI model is not available right now. I can still help with basic finance operations like logging expenses and setting budgets."
    llm_error_message: str = "I'm having trouble processing that request right now. Please try rephrasing or ask about your expenses or budgets."
    
    # Error message templates
    amount_parse_error_template: str = "I couldn't find the amount. Try like: I spent {currency_symbol}12 on {default_category}."
    budget_parse_error_template: str = "I couldn't find the budget amount. Try: Set budget for {category} to {currency_symbol}200."
    no_expenses_message_template: str = "No expenses logged yet. Try: I spent {currency_symbol}8 on {category}."
    
    # Response templates
    balance_response_template: str = "Your current balance is {currency_symbol}{balance}."
    expense_logged_template: str = "Logged {currency_symbol}{amount} for {category}. New balance: {currency_symbol}{balance}.{warnings}"
    budget_set_template: str = "Budget set for {category}: {currency_symbol}{amount}."
    budget_warning_template: str = "You are close to your {category} budget ({currency_symbol}{limit_amount}). Spent: {currency_symbol}{spent}."
    budget_reached_template: str = "You reached your {category} budget ({currency_symbol}{limit_amount}). Spent: {currency_symbol}{spent}."
    summary_header_template: str = "Spending summary: {spending_breakdown} Balance: {currency_symbol}{balance}{warnings}"

    # Regex patterns
    amount_regex_pattern: str = r'(\$?\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\$?)'
    category_regex_pattern: str = r'(?:on|for)\s+([A-Za-z\-\_]+)'
    set_budget_regex_pattern: str = r'(?:set|update)\s+(?:a\s+)?budget(?:\s+limit)?\s*(?:for)?\s*([A-Za-z\-\_]+)?\s*(?:to|=)?\s*(\$?\d+(?:\.\d{1,2})?)'

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Initialize settings with validation
try:
    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
    logger = logging.getLogger(__name__)
    logger.info("✓ Settings loaded successfully from .env")
except Exception as e:
    print(f"❌ ERROR: Failed to load settings from .env file")
    print(f"   Make sure you have a .env file with all required variables")
    print(f"   Error details: {e}")
    raise

# =========================
# DATABASE SETUP
# =========================
DATABASE_URL = f"{settings.db_driver}://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"
try:
    DB_ENGINE = create_engine(DATABASE_URL, echo=settings.db_echo)
    with DB_ENGINE.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Connected to PostgreSQL")
except Exception as e:
    logger.warning(f"PostgreSQL unavailable: {e}. Using SQLite for demo.")
    DATABASE_URL = "sqlite:///finance_demo.db"
    DB_ENGINE = create_engine(DATABASE_URL, echo=settings.db_echo)

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=DB_ENGINE)

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# DEFINE Transaction CLASS (BEFORE using it in functions)
class Transaction(Base):
    __tablename__ = 'transactions'
    transaction_id = Column(String, primary_key=True, name='Transaction ID', quote=True)
    date = Column(String, name='Date', quote=True)
    currency = Column(String, name='Currency', quote=True)
    sender = Column(String, name='Sender', quote=True)
    receiver = Column(String, name='Receiver', quote=True)
    amount = Column(String, name='Amount', quote=True)
    fee = Column(String, name='Fee', quote=True)
    type = Column(String, name='Type', quote=True)

# Create tables
Base.metadata.create_all(bind=DB_ENGINE)

# =========================
# LLM CHECK
# =========================
def initialize_gemini():
    if not settings.gemini_enabled:
        logger.info("Gemini disabled")
        return False
    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        logger.warning("Gemini API key not set")
        return False
    logger.info("Gemini configured")
    return True

llm_available = initialize_gemini()

# =========================
# FASTAPI APP
# =========================
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info(f"Starting {settings.app_title}")
    yield
    logger.info(f"Shutting down {settings.app_title}")

app = FastAPI(title=settings.app_title, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_history: Dict[str, List[Dict[str, str]]] = defaultdict(list)
user_budgets: Dict[str, Dict[str, float]] = defaultdict(dict)
user_budgets["demo_user"] = {"food": 500.0, "entertainment": 200.0, "transport": 300.0}
user_budgets["anon"] = {"food": 400.0, "entertainment": 150.0, "transport": 250.0}

# Regex utility
class RegexPatterns:
    def __init__(self, settings_obj):
        self.amount_re = re.compile(settings_obj.amount_regex_pattern)
        self.category_re = re.compile(settings_obj.category_regex_pattern, re.IGNORECASE)
        self.set_budget_re = re.compile(settings_obj.set_budget_regex_pattern, re.IGNORECASE)

patterns = RegexPatterns(settings)

# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    user_id: Optional[str] = settings.default_user_id
    message: str

class ChatResponse(BaseModel):
    intent: str
    entities: Dict[str, Any]
    reply: str
    balance: float
    budgets: Dict[str, float]
    recent_expenses: List[Dict[str, Any]]

# =========================
# DATABASE FUNCTIONS (NOW Transaction is defined)
# =========================
def add_transaction(db: Session, user_id: str, transaction_id: str, date: datetime, 
                    currency: str, sender: str, receiver: str, amount: float, 
                    fee: float = 0.0, transaction_type: str = "expense") -> Transaction:
    transaction = Transaction(
        transaction_id=transaction_id,
        date=date.isoformat(),
        currency=currency,
        sender=sender,
        receiver=receiver,
        amount=str(amount),
        fee=str(fee),
        type=transaction_type
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    return transaction

def get_user_transactions(db: Session, user_id: str, limit: int = None) -> List[Transaction]:
    query = db.query(Transaction).filter(Transaction.sender == user_id).order_by(Transaction.date.desc())
    if limit:
        query = query.limit(limit)
    return query.all()

def compute_balance(db: Session, user_id: str) -> float:
    transactions = get_user_transactions(db, user_id)
    balance = settings.default_starting_balance
    for tx in transactions:
        amount = float(tx.amount or 0.0)
        balance += amount if tx.type == "income" else -amount
    return round(balance, 2)

def category_spend(db: Session, user_id: str) -> Dict[str, float]:
    transactions = get_user_transactions(db, user_id)
    spending = defaultdict(float)
    for tx in transactions:
        if tx.type == "expense":
            spending[tx.receiver or "general"] += float(tx.amount or 0.0)
    return dict(spending)

def budget_warnings(user_id: str, db: Session) -> List[str]:
    warnings = []
    expense_totals = category_spend(db, user_id)
    budgets = user_budgets.get(user_id, {})
    for category, limit in budgets.items():
        spent = expense_totals.get(category, 0.0)
        if spent >= limit:
            warnings.append(f"⚠️ You reached your {category} budget ({settings.currency_symbol}{limit}). Spent: {settings.currency_symbol}{spent}.")
        elif spent >= settings.budget_warning_threshold * limit:
            warnings.append(f"⚠️ Close to {category} budget ({settings.currency_symbol}{limit}). Spent: {settings.currency_symbol}{spent}.")
    return warnings

# =========================
# PARSING & INTENT
# =========================
def parse_float_amount(text: str) -> Optional[float]:
    m = patterns.amount_re.search(text.replace(",", ""))
    if not m: return None
    val = m.group(1)
    for s in settings.currency_symbols:
        val = val.replace(s, "")
    try: return float(val.strip())
    except: return None

def parse_category(text: str) -> Optional[str]:
    m = patterns.category_re.search(text.lower())
    return m.group(1).lower() if m else None

def detect_intent(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["balance", "how much money"]): return "check_balance"
    if any(k in t for k in ["spent", "purchase", "paid"]) and parse_float_amount(t) is not None: return "log_expense"
    if "budget" in t and any(k in t for k in ["set", "update", "change"]): return "set_budget"
    if any(k in t for k in ["summary", "report", "breakdown"]): return "spending_summary"
    return "smalltalk_or_qa"

# =========================
# GEMINI API
# =========================
async def call_gemini_api(messages: List[Dict[str, str]]) -> str:
    contents = [{"parts": [{"text": msg["content"]}], "role": "model" if msg["role"]=="assistant" else None} for msg in messages]
    payload = {"contents": contents, "generationConfig": {
        "maxOutputTokens": settings.llm_max_tokens,
        "temperature": settings.llm_temperature,
        "topK": settings.llm_top_k,
        "topP": settings.llm_top_p
    }}
    url = f"{settings.gemini_api_url}?key={settings.gemini_api_key}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers={"Content-Type": "application/json"}, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "candidates" in data and data["candidates"]: 
            return data["candidates"][0]["content"]["parts"][0]["text"]
        return "LLM failed to generate a response."

async def llm_answer(user_id: str, user_text: str) -> str:
    if not llm_available: return settings.llm_unavailable_message
    try:
        messages = [{"role": "system", "content": settings.system_prompt}] + chat_history[user_id][-settings.recent_chat_context:]
        messages.append({"role": "user", "content": user_text})
        reply = await call_gemini_api(messages)
        chat_history[user_id].append({"role": "user", "content": user_text})
        chat_history[user_id].append({"role": "assistant", "content": reply.strip()})
        chat_history[user_id] = chat_history[user_id][-settings.max_chat_history:]
        return reply.strip()
    except Exception as e:
        logger.error(e)
        return settings.llm_error_message

# =========================
# API ENDPOINTS
# =========================
@app.get("/")
def root(): return {"message": "Finley Finance Assistant API", "status": "running", "version": "1.0.0"}

@app.get("/health")
def health():
    db_status = "connected"
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {e}"
    return {"ok": True, "database": db_status, "llm": "available" if llm_available else "not_available"}

@app.get("/users/{user_id}/stats")
def user_stats(user_id: str):
    with get_db() as db:
        transactions = get_user_transactions(db, user_id)
        return {
            "user_id": user_id,
            "balance": compute_balance(db, user_id),
            "total_transactions": len(transactions),
            "category_spending": category_spend(db, user_id),
            "budgets": user_budgets.get(user_id, {}),
            "recent_transactions": [{"amount": t.amount,"type": t.type,"sender": t.sender,"receiver": t.receiver,"date": t.date} for t in transactions[:settings.max_expenses_display]]
        }

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    user_id = req.user_id or settings.default_user_id
    text = req.message.strip()
    with get_db() as db:
        intent = detect_intent(text)
        entities, reply = {}, ""
        if intent == "check_balance":
            bal = compute_balance(db, user_id)
            reply = f"Your balance is {settings.currency_symbol}{bal}."
        elif intent == "log_expense":
            amount = parse_float_amount(text)
            category = parse_category(text) or settings.default_category
            if amount is None:
                reply = settings.amount_parse_error_template.format(currency_symbol=settings.currency_symbol, default_category=settings.default_category)
            else:
                transaction_id = str(uuid.uuid4())
                add_transaction(db, user_id, transaction_id, datetime.now(timezone.utc), settings.currency_symbol, user_id, category, amount)
                bal = compute_balance(db, user_id)
                warn_txt = "\n".join(budget_warnings(user_id, db))
                reply = f"Logged {settings.currency_symbol}{amount} for {category}. Balance: {settings.currency_symbol}{bal}\n{warn_txt}"
                entities = {"amount": amount, "category": category}
        elif intent == "set_budget":
            m = patterns.set_budget_re.search(text)
            amount = parse_float_amount(text)
            category = parse_category(text) or settings.default_category
            if amount is None:
                reply = settings.budget_parse_error_template.format(default_category=settings.default_category, currency_symbol=settings.currency_symbol)
            else:
                user_budgets[user_id][category] = amount
                entities = {"amount": amount, "category": category}
                reply = f"✅ Budget set for {category}: {settings.currency_symbol}{amount}"
        elif intent == "spending_summary":
            totals = category_spend(db, user_id)
            if not totals:
                reply = settings.no_expenses_message_template.format(currency_symbol=settings.currency_symbol, default_category=settings.default_category)
            else:
                lines = [f"- {cat}: {settings.currency_symbol}{amt:.2f}" for cat, amt in totals.items()]
                bal = compute_balance(db, user_id)
                warn_txt = "\n".join(budget_warnings(user_id, db))
                reply = f"**Spending summary:**\n" + "\n".join(lines) + f"\nBalance: {settings.currency_symbol}{bal}\n{warn_txt}"
        else:
            reply = await llm_answer(user_id, text)
        bal = compute_balance(db, user_id)
        recent = [{"amount": t.amount, "type": t.type, "sender": t.sender, "receiver": t.receiver, "date": t.date} for t in get_user_transactions(db, user_id, settings.recent_expenses_limit)]
        return ChatResponse(intent=intent, entities=entities, reply=reply, balance=bal, budgets=user_budgets.get(user_id, {}), recent_expenses=recent)

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("localGPT:app", host=settings.app_host, port=settings.app_port, reload=True)