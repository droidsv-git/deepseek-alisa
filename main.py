import os
import asyncio
import requests
from fastapi import FastAPI, Request, BackgroundTasks

app = FastAPI()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GPT_API_KEY = os.getenv("GPT_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"

GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")
GITHUB_API_URL = "https://models.github.ai/inference/chat/completions"

RENDER_URL = "https://deepseek-alisa-fusq.onrender.com"

# Твой оригинальный кэш
CACHE = {}
USER_MODELS = {}

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
    """Твоя оригинальная функция нарезки текста"""
    if len(text) <= size:
        return [text]
    
    chunks = []
    while len(text) > 0:
        if len(text) <= size:
            chunks.append(text)
            break
        split_idx = text.rfind('\n', 0, size)
        if split_idx == -1:
            split_idx = size
        chunks.append(text[:split_idx])
        text = text[split_idx:].lstrip()
    return chunks

def send_groq_request(model_path: str, user_text: str, timeout_val: int):
    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {GPT_API_KEY}"},
        json={
            "model": model_path,
            "messages": [
                {
                    "role": "system",
                    "content": "Отвечай на русском языке. Не используй Markdown. Не показывай внутренние размышления."
                },
                {"role": "user", "content": user_text}
            ]
        },
        timeout=timeout_val
    )
    data = response.json()
    if "choices" not in data:
        raise Exception(f"Groq API Error: {data}")
    message = data["choices"][0]["message"]
    answer = message.get("content") or message.get("reasoning") or message.get("reasoning_content") or ""
    return answer.strip()

def send_gemini_request(user_text: str, timeout_val: int):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    # Без ограничений токенов
    response = requests.post(
        url,
        json={"contents": [{"parts": [{"text": user_text}]}]},
        timeout=timeout_val
    )
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        raise Exception(f"Gemini API Error: {data}")

def send_github_request(user_text: str, timeout_val: int):
    if not GITHUB_API_KEY:
        raise Exception("Не задан GITHUB_API_KEY.")
    response = requests.post(
        GITHUB_API_URL,
        headers={
            "Authorization": f"Bearer {GITHUB_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json"
        },
        json={
            "model": "deepseek/DeepSeek-R1",
            "messages": [
                {
                    "role": "system",
                    "content": "Отвечай на русском языке. Не используй Markdown. Не показывай внутренние размышления."
                },
                {"role": "user", "content": user_text}
            ]
            # Убрано "max_tokens": 40, чтобы не обрубало текст
        },
        timeout=timeout_val
    )
    data = response.json()
    if "choices" not in data:
        raise Exception(f"GitHub Models API Error: {data}")
    message = data["choices"][0]["message"]
    answer = message.get("content") or message.get("reasoning_content") or ""
    return answer.strip()

def send_request_to_provider(model_key: str, user_text: str, timeout_val: int = 30):
    if model_key == "джипити":
        return send_groq_request("openai/gpt-oss-120b", user_text, timeout_val)
    if model_key == "джемини":
        return send_gemini_request(user_text, timeout_val)
    if model_key == "дипсик":
        return send_github_request(user_text, timeout_val)
    raise Exception("Неизвестная модель.")

def get_intro_prompt(model_key: str):
    if model_key == "джипити":
        return "Поздоровайся с Сергеем по имени, как старый друг. Скажи что ты Джипити (GPT-OSS 120B). Только одно короткое предложение."
    if model_key == "джемини":
        return "Поздоровайся с Сергеем по имени, как старый друг. Скажи что ты Джемини (Gemini Flash от Google). Только одно короткое предложение."
    if model_key == "дипсик":
        return "Поздоровайся с Сергеем по имени, как старый друг. Скажи что ты Дипсик (DeepSeek). Только одно короткое предложение."

def fetch_background(session_id: str, user_text: str, model_key: str):
    try:
        answer = send_request_to_provider(model_key, user_text, timeout_val=45)
        chunks = split_text(answer, 900)
        CACHE[session_id] = chunks
    except Exception as e:
        CACHE[session_id] = [f"Ошибка генерации: {str(e)[:150]}"]

@app.post("/")
async def main(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    req_data = body.get("request", {})
    user_text = (req_data.get("original_utterance") or req_data.get("command") or "").lower().strip()
    session_id = body["session"]["session_id"]
    is_new_session = body.get("session", {}).get("new", False)

    current_model_key = USER_MODELS.get(session_id, "джипити")

    # 1. Меню
    if is_new_session or user_text in ["меню", "модели", "какая модель"]:
        menu_text = (
            "Кибер Ноотроп запущен. Подключенные модели: "
            "Джипити — GPT-OSS 120B. "
            "Джемини — Gemini Flash. "
            "Дипсик — DeepSeek. "
            "Вернуть Алису. "
            "Какую модель подключить?"
        )
        return {
            "version": body["version"], "session": body["session"],
            "response": {"end_session": False, "text": menu_text, "tts": menu_text}
        }

    # 2. Сброс
    if user_text in ["вернуть алису", "вернуться к алисе", "сбросить модель", "отключить модель", "по умолчанию"]:
        if session_id in USER_MODELS:
            del USER_MODELS[session_id]
        return {
            "version": body["version"], "session": body["session"],
            "response": {"end_session": True, "text": "Возврат к стандартной Алисе.", "tts": "Возврат к стандартной Алисе."}
        }

    # 3. Выбор модели
    found_model_key = None
    model_aliases = {
        "джипити": ["джипити", "gpt", "гпт", "джи пи ти"],
        "джемини": ["джемини", "gemini", "джемени"],
        "дипсик": ["дипсик", "deepseek", "дипсиг", "дип сик"]
    }
    for model_key, aliases in model_aliases.items():
        for alias in aliases:
            if alias in user_text:
                found_model_key = model_key
                break
        if found_model_key:
            break

    if found_model_key:
        USER_MODELS[session_id] = found_model_key
        if session_id in CACHE:
            del CACHE[session_id]

        intro_prompt = get_intro_prompt(found_model_key)
        try:
            # Таймаут 4.0 секунды для приветствия
            greeting = send_request_to_provider(found_model_key, intro_prompt, timeout_val=4.0)
            answer = f"{greeting} Чтобы вернуться к списку моделей, скажите «меню». Чтобы вернуться к Алисе, скажите «вернуть алису»."
            return {
                "version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": answer, "tts": answer}
            }
        except Exception:
            CACHE[session_id] = "PENDING"
            background_tasks.add_task(fetch_background, session_id, intro_prompt, found_model_key)
            answer = f"Подключаю модель {found_model_key.capitalize()}. Она формирует приветствие. Скажите «готово», чтобы услышать его."
            return {
                "version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": answer, "tts": answer}
            }

    # 4. Твоя оригинальная обработка «дальше»
    if user_text in ["дальше", "дальше?"]:
        if session_id in CACHE and CACHE[session_id] != "PENDING":
            chunks = CACHE[session_id]
            if chunks:
                current_chunk = chunks.pop(0)
                if not chunks:
                    del CACHE[session_id]
                else:
                    current_chunk += "\n\nСкажи «дальше», чтобы послушать продолжение."
                return {
                    "version": body["version"], "session": body["session"],
                    "response": {"end_session": False, "text": current_chunk, "tts": current_chunk}
                }
        answer = "Нет сохраненного продолжения. Задай новый вопрос."
        return {
            "version": body["version"], "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }

    # 5. Твоя оригинальная обработка «готово»
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
                        current_chunk += "\n\nСкажи «дальше», чтобы послушать продолжение."
                    answer = current_chunk
                else:
                    answer = "Кэш пуст."
                    del CACHE[session_id]
        else:
            answer = "Нет активных фоновых задач."
        return {
            "version": body["version"], "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }

    # 6. Новый запрос
    if session_id in CACHE:
        del CACHE[session_id]

    try:
        # Таймаут 4.0 секунды
        answer = send_request_to_provider(current_model_key, user_text, timeout_val=4.0)

        if len(answer) > 900:
            chunks = split_text(answer, 900)
            first_chunk = chunks.pop(0)
            CACHE[session_id] = chunks
            first_chunk += "\n\nОтвет длинный, скажи «дальше», чтобы продолжить."
            answer = first_chunk

        return {
            "version": body["version"], "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }

    except Exception:
        CACHE[session_id] = "PENDING"
        background_tasks.add_task(fetch_background, session_id, user_text, current_model_key)
        answer = "Вопрос сложный, обдумываю в фоне. Спроси «Готово?» через пару секунд."
        return {
            "version": body["version"], "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }
