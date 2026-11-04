import os
import httpx
from fastapi import FastAPI, Request

app = FastAPI()

DEEPSEEK_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

@app.get("/")
async def root():
    # Эндпоинт для пинга (UptimeRobot), чтобы Render не засыпал
    return {"status": "ok"}

@app.post("/")
async def main(request: Request):
    body = await request.json()
    
    # Обработка пустого ввода при первом пинге от Яндекса
    request_data = body.get("request", {})
    user_text = request_data.get("original_utterance", "")
    
    if not user_text:
        user_text = "Привет"

    answer = "Извини, не успел сформировать ответ. Попробуй еще раз."

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",  # Очень быстрая модель на Groq
        "messages": [
            {"role": "system", "content": "Отвечай кратко, емко и понятными фразами для голосового помощника Алиса. Максимум 2-3 предложения."},
            {"role": "user", "content": user_text}
        ],
        "max_tokens": 150  # Ограничение длины ускоряет генерацию
    }

    try:
        # Асинхронный запрос с жестким тайм-аутом в 2.5 секунды
        async with httpx.AsyncClient(timeout=2.5) as client:
            res = await client.post(DEEPSEEK_API_URL, headers=headers, json=payload)
            if res.status_code == 200:
                data = res.json()
                answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error during API call: {e}")

    return {
        "version": body.get("version", "1.0"),
        "session": body.get("session", {}),
        "response": {
            "end_session": False,
            "text": answer
        }
    }
