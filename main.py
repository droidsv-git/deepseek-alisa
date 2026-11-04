import os
import asyncio
import requests
from fastapi import FastAPI, Request

app = FastAPI()

DEEPSEEK_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Твой реальный публичный адрес на Render для само-пинга
RENDER_URL = "https://deepseek-alisa-fusq.onrender.com"

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

@app.post("/")
async def main(request: Request):
    body = await request.json()
    req_data = body.get("request", {})
    user_text = req_data.get("original_utterance") or req_data.get("command") or ""

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": user_text}],
            },
            timeout=25
        )
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"Ошибка запроса: {e}"

    # Жесткое ограничение Яндекса: ответ не должен превышать 1024 символа
    if len(answer) > 1000:
        answer = answer[:995] + "..."

    return {
        "version": body["version"],
        "session": body["session"],
        "response": {
            "end_session": False,
            "text": answer,
            "tts": answer
        }
    }
