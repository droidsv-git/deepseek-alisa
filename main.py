import os
import asyncio
import requests
from fastapi import FastAPI, Request, BackgroundTasks

app = FastAPI()

DEEPSEEK_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

RENDER_URL = "https://deepseek-alisa-fusq.onrender.com"

# Кэш для хранения порций ответов по session_id
CACHE = {}

async def self_ping():
    """Каждые 10 минут пингует сам себя, чтобы Render не засыпал"""
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
    return [text[i:i+size] for i in range(0, len(text), size)]

def fetch_background(session_id: str, user_text: str):
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

    # 1. Продолжение — строго по слову «дальше»
    if user_text in ["дальше", "дальше?"]:
        if session_id in CACHE and CACHE[session_id] != "PENDING":
            chunks = CACHE[session_id]
            if chunks:
                current_chunk = chunks.pop(0)
                if not chunks:
                    del CACHE[session_id]
                else:
                    current_chunk += " Скажи «дальше», чтобы послушать продолжение."
                return {
                    "version": body["version"],
                    "session": body["session"],
                    "response": {"end_session": False, "text": current_chunk, "tts": current_chunk}
                }
        answer = "Нет сохраненного продолжения. Задай новый вопрос."
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }

    # 2. Вывод из кэша / проверка статуса — строго по слову «готово»
    if user_text in ["готово", "готово?"]:
        if session_id in CACHE:
            status = CACHE[session_id]
            if status == "PENDING":
                answer = "Еще думаю над ответом, спроси «Готово?» через секунду."
            else:
                chunks = status
                if chunks:
                    current_chunk = chunks.pop(0)
                    if not chunks:
                        del CACHE[session_id]
                    else:
                        current_chunk += " Скажи «дальше», чтобы послушать продолжение."
                    answer = current_chunk
                else:
                    answer = "Кэш пуст."
                    del CACHE[session_id]
        else:
            answer = "Нет активных фоновых задач."
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }

    # 3. Любое другое слово — стираем старый кэш и считаем это новым запросом
    if session_id in CACHE:
        del CACHE[session_id]

    # Синхронный запрос в пределах лимита Яндекса (2.7 сек)
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": user_text}],
            },
            timeout=2.7
        )
        data = response.json()
        answer = data["choices"][0]["message"]["content"]

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
        # Если думает дольше 2.7 сек — уходим в фон, отдаем заглушку
        CACHE[session_id] = "PENDING"
        background_tasks.add_task(fetch_background, session_id, user_text)
        
        answer = "Вопрос сложный, обдумываю в фоне. Спроси «Готово?» через пару секунд."
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }
