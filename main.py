import os
import asyncio
import requests
from fastapi import FastAPI, Request

app = FastAPI()

DEEPSEEK_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

RENDER_URL = "https://deepseek-alisa-fusq.onrender.com"

# Хранилище для длинных ответов, разбитых на части (для диалога по шагам)
SESSION_CHUNKS = {}

async def self_ping():
    while True:
        await asyncio.sleep(600)
        try:
            requests.get(RENDER_URL, timeout=5)
        except Exception:
            pass

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(self_ping())

@app.get("/")
async def root():
    return {"status": "awake"}

def split_text(text: str, size: int = 900):
    """Режет длинный текст на куски, безопасные для лимита Яндекса"""
    return [text[i:i+size] for i in range(0, len(text), size)]

@app.post("/")
async def main(request: Request):
    body = await request.json()
    req_data = body.get("request", {})
    user_text = (req_data.get("original_utterance") or req_data.get("command") or "").lower().strip()
    session_id = body["session"]["session_id"]

    # Если пользователь просит продолжить предыдущий ответ
    if user_text in ["дальше", "продолжи", "продолжить", "еще"] and session_id in SESSION_CHUNKS:
        chunks = SESSION_CHUNKS[session_id]
        if chunks:
            current_chunk = chunks.pop(0) # Берем следующую часть
            if not chunks:
                del SESSION_CHUNKS[session_id] # Если части кончились, удаляем сессию
            return {
                "version": body["version"],
                "session": body["session"],
                "response": {
                    "end_session": False,
                    "text": current_chunk,
                    "tts": current_chunk
                }
            }

    # Новый запрос к модели
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": user_text}],
            },
            timeout=2.8 # Укладываемся в тайм-аут Яндекса на первый ответ
        )
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
    except Exception:
        answer = "Не успеваю ответить за отведенное время, попробуй еще раз."

    # Если текст длиннее 900 символов — режем на части
    if len(answer) > 900:
        chunks = split_text(answer, 900)
        first_chunk = chunks.pop(0)
        
        # Сохраняем оставшиеся части в память
        SESSION_CHUNKS[session_id] = chunks
        
        # Дописываем подсказку для пользователя
        first_chunk += " Ответ длинный. Скажи «дальше», чтобы послушать продолжение."
        tts_text = first_chunk
    else:
        tts_text = answer

    return {
        "version": body["version"],
        "session": body["session"],
        "response": {
            "end_session": False,
            "text": tts_text,
            "tts": tts_text
        }
    }
