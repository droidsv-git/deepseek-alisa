import os
import asyncio
import requests
from fastapi import FastAPI, Request, BackgroundTasks

app = FastAPI()

DEEPSEEK_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Твой реальный публичный адрес на Render для само-пинга
RENDER_URL = "https://deepseek-alisa-fusq.onrender.com"

# Кэш для хранения порций ответов по session_id
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
    # Запускаем фоновую задачу само-пинга при старте приложения
    asyncio.create_task(self_ping())

@app.get("/")
async def root():
    return {"status": "awake"}

def split_text(text: str, size: int = 900):
    """Режет длинный текст на порции для голосовой выдачи"""
    return [text[i:i+size] for i in range(0, len(text), size)]

def fetch_background(session_id: str, user_text: str):
    """Фоновый досыл запроса, если синхронный не уложился в 3 секунды"""
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
        chunks = split_text(answer, 900)
        CACHE[session_id] = chunks
    except Exception as e:
        CACHE[session_id] = [f"Ошибка генерации: {e}"]

@app.post("/")
async def main(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    req_data = body.get("request", {})
    user_text = (req_data.get("original_utterance") or req_data.get("command") or "").lower().strip()
    session_id = body["session"]["session_id"]

    # 1. Обработка стоп-слов
    stop_words = ["хватит", "стоп", "остановись", "завязывай"]
    if any(word in user_text for word in stop_words):
        if session_id in CACHE:
            del CACHE[session_id]
        answer = "Остановлено."
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }

    # 2. Если в кэше есть готовые порции — отдаем следующую
    if session_id in CACHE and CACHE[session_id] != "PENDING":
        chunks = CACHE[session_id]
        if chunks:
            current_chunk = chunks.pop(0)
            if not chunks:
                del CACHE[session_id]
            if chunks:
                current_chunk += " Скажи что угодно, чтобы послушать продолжение."
            return {
                "version": body["version"],
                "session": body["session"],
                "response": {"end_session": False, "text": current_chunk, "tts": current_chunk}
            }

    # 3. Если в кэше висит статус PENDING (фоновый процесс еще думает)
    if session_id in CACHE and CACHE[session_id] == "PENDING":
        answer = "Еще думаю над ответом, спроси «Готово?» через секунду."
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }

    # 4. Пробуем получить ответ синхронно, укладываясь в лимит Яндекса (~2.7 секунды)
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": user_text}],
            },
            timeout=2.7  # Строго меньше 3 секунд Яндекса
        )
        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        # Если ответ длинный — режем на порции
        if len(answer) > 900:
            chunks = split_text(answer, 900)
            first_chunk = chunks.pop(0)
            CACHE[session_id] = chunks
            first_chunk += " Ответ длинный, скажи «дальше», чтобы продолжить."
            answer = first_chunk

        return {
            "version": body["version"],
            "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }

    except Exception:
        # Если вылетели по таймауту (>2.7с), ставим заглушку и уходим в фон
        CACHE[session_id] = "PENDING"
        background_tasks.add_task(fetch_background, session_id, user_text)
        
        answer = "Вопрос сложный, я обдумываю его в фоне. Спроси «Готово?» через пару секунд."
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }
