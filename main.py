import os
import asyncio
import requests
from fastapi import FastAPI, Request, BackgroundTasks

app = FastAPI()

DEEPSEEK_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Твой реальный публичный адрес на Render для само-пинга
RENDER_URL = "https://deepseek-alisa-fusq.onrender.com"

# Кэш для фоновых ответов, чтобы Яндекс не падал по таймауту
CACHE = {}

async def self_ping():
    """Каждые 10 минут пингует сам себя, чтобы Render не засыпал"""
    while True:
        await asyncio.sleep(600)  # 600 секунд = 10 минут
        try:
            requests.get(RENDER_URL, timeout=5)
        except Exception:
            pass

@app.on_event("startup")
async def startup_event():
    # Запускаем фоновую задачу само-пинга при старте
    asyncio.create_task(self_ping())

@app.get("/")
async def root():
    return {"status": "awake"}

def fetch_groq_background(session_id: str, user_text: str):
    """Фоновый запрос к Groq, не ограниченный 3 секундами Яндекса"""
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": user_text}],
            },
            timeout=30
        )
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        CACHE[session_id] = answer
    except Exception as e:
        CACHE[session_id] = f"Ошибка генерации: {e}"

@app.post("/")
async def main(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    req_data = body.get("request", {})
    user_text = req_data.get("original_utterance") or req_data.get("command") or ""
    session_id = body["session"]["session_id"]

    # 1. Если ответ для этой сессии уже сгенерирован в фоне — отдаем его Алисе
    if session_id in CACHE and CACHE[session_id] != "PENDING":
        answer = CACHE.pop(session_id)
    
    # 2. Если он еще генерируется
    elif session_id in CACHE and CACHE[session_id] == "PENDING":
        answer = "Секунду, еще дописываю ответ, повтори запрос."

    # 3. Первый запрос: запускаем фонового воркера и мгновенно отвечаем Яндексу
    else:
        CACHE[session_id] = "PENDING"
        background_tasks.add_task(fetch_groq_background, session_id, user_text)
        answer = "Минутку, размышляю над ответом... Спроси «Что там?» через пару секунд."

    return {
        "version": body["version"],
        "session": body["session"],
        "response": {
            "end_session": False,
            "text": answer
        }
    }
