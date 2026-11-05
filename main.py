import os
import asyncio
import requests
from fastapi import FastAPI, Request, BackgroundTasks

app = FastAPI()

# Прямое подключение для Джипити (Groq)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GPT_API_KEY = os.getenv("GPT_API_KEY")

# Подключение через OpenRouter для остальных моделей
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

RENDER_URL = "https://deepseek-alisa-fusq.onrender.com"

# Кэш для хранения порций ответов по session_id
CACHE = {}

# Хранилище выбранной модели для каждого сеанса пользователя
USER_MODELS = {}

# Каталог моделей (команды без слова «включи»)
AVAILABLE_MODELS = {
    "джипити": {
        "aliases": ["джипити", "gpt", "гпт", "джи пи ти"],
        "provider": "groq",
        "path": "openai/gpt-oss-120b",
        "name": "GPT-OSS 120B",
        "greeting": "Здравствуйте! Я модель Джипити GPT-OSS 120B, готова к работе."
    },
    "джемини": {
        "aliases": ["джемини", "gemini", "джемени"],
        "provider": "openrouter",
        "path": "google/gemini-2.5-flash:free",
        "name": "Gemini 2.5 Flash",
        "greeting": "Приветствую! Я модель Джемини 2.5 Flash, слушаю вас."
    },
    "дипсик": {
        "aliases": ["дипсик", "deepseek", "дипсиг", "дип сик"],
        "provider": "openrouter",
        "path": "deepseek/deepseek-v4-flash:free",
        "name": "DeepSeek V4 Flash",
        "greeting": "Здравствуйте! Я модель Дипсик, задавайте ваш вопрос."
    }
}

DEFAULT_MODEL_KEY = "джипити"

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
    """Аккуратно режет текст по строкам, не ломая код и разметку посередине"""
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

def send_request_to_provider(model_key: str, user_text: str, timeout_val: int = 30):
    """Отправляет запрос нужному провайдеру"""
    m_info = AVAILABLE_MODELS[model_key]
    provider = m_info["provider"]
    model_path = m_info["path"]

    if provider == "groq":
        return requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GPT_API_KEY}"},
            json={
                "model": model_path,
                "messages": [{"role": "user", "content": user_text}],
            },
            timeout=timeout_val
        )
    else:
        return requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": RENDER_URL,
                "X-Title": "Cyber Nootrop Alice"
            },
            json={
                "model": model_path,
                "messages": [{"role": "user", "content": user_text}],
            },
            timeout=timeout_val
        )

def fetch_background(session_id: str, user_text: str, model_key: str):
    try:
        response = send_request_to_provider(model_key, user_text, timeout_val=30)
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
    is_new_session = body.get("session", {}).get("new", False)

    # Текущая модель для сессии
    current_model_key = USER_MODELS.get(session_id, DEFAULT_MODEL_KEY)

    # 0. Структурированное голосовое меню (без слова «включи»)
    if is_new_session or user_text in ["меню", "модели", "какая модель"]:
        menu_text = (
            "Кибер Ноотроп запущен. Подключенные модели: "
            "1. Джипити — модель GPT-OSS 120B. "
            "2. Джемини — модель Gemini 2.5 Flash. "
            "3. Дипсик — модель DeepSeek V4 Flash. "
            "4. Вернуться к Алисе. "
            "Какую модель подключить?"
        )
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {"end_session": False, "text": menu_text, "tts": menu_text}
        }

    # Обработка возврата к Алисе
    if user_text in ["сбросить модель", "отключить модель", "по умолчанию", "вернуться к алисе"]:
        if session_id in USER_MODELS:
            del USER_MODELS[session_id]
        answer = "Возврат к стандартной Алисе."
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {"end_session": True, "text": answer, "tts": answer}
        }

    # Прямое переключение по имени модели (без обязательного слова «включи»)
    found_model_key = None
    for m_key, m_data in AVAILABLE_MODELS.items():
        for alias in m_data["aliases"]:
            if alias in user_text:
                found_model_key = m_key
                break
        if found_model_key:
            break

    if found_model_key:
        USER_MODELS[session_id] = found_model_key
        model_greeting = AVAILABLE_MODELS[found_model_key]["greeting"]
        answer = f"Чтобы вернуться к списку моделей, скажите «меню». Чтобы вернуться к Алисе, скажите «вернуться к алисе». {model_greeting}"
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }

    # 1. Продолжение — строго по слову «дальше»
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
                        current_chunk += "\n\nСкажи «дальше», чтобы послушать продолжение."
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
        response = send_request_to_provider(current_model_key, user_text, timeout_val=2.7)
        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        if len(answer) > 900:
            chunks = split_text(answer, 900)
            first_chunk = chunks.pop(0)
            CACHE[session_id] = chunks
            first_chunk += "\n\nОтвет длинный, скажи «дальше», чтобы продолжить."
            answer = first_chunk

        return {
            "version": body["version"],
            "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }

    except Exception:
        # Если думает дольше 2.7 сек — уходим в фон, отдаем заглушку
        CACHE[session_id] = "PENDING"
        background_tasks.add_task(fetch_background, session_id, user_text, current_model_key)
        
        answer = "Вопрос сложный, обдумываю в фоне. Спроси «Готово?» через пару секунд."
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }
