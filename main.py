import os
import asyncio
import requests
import time
from fastapi import FastAPI, Request, BackgroundTasks

app = FastAPI()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GPT_API_KEY = os.getenv("GPT_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"

GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")
GITHUB_API_URL = "https://models.github.ai/inference/chat/completions"

RENDER_URL = "https://deepseek-alisa-fusq.onrender.com"

CACHE = {}
USER_MODELS = {}
HISTORY = {}

SESSION_TTL = 1800

def get_history(user_id: str):
    if user_id in HISTORY and time.time() - HISTORY[user_id]["last_update"] < SESSION_TTL:
        return HISTORY[user_id]["data"]
    return []

def update_history(user_id: str, role: str, content: str):
    if user_id not in HISTORY:
        HISTORY[user_id] = {"data": [], "last_update": time.time()}
    HISTORY[user_id]["data"].append({"role": role, "content": content})
    HISTORY[user_id]["last_update"] = time.time()
    if len(HISTORY[user_id]["data"]) > 20:
        HISTORY[user_id]["data"] = HISTORY[user_id]["data"][-20:]

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

# --- БАЗОВЫЕ ЗАПРОСЫ ---

def send_groq_request(model_path: str, user_text: str, timeout_val: int, history: list = None):
    messages = [{"role": "system", "content": "Отвечай на русском языке. Не используй Markdown. Отвечай кратко."}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    
    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {GPT_API_KEY}"},
        json={"model": model_path, "messages": messages},
        timeout=timeout_val
    )
    data = response.json()
    if "choices" not in data: raise Exception("Ошибка Groq.")
    message = data["choices"][0]["message"]
    return (message.get("content") or message.get("reasoning") or "").strip()

def send_gemini_request(user_text: str, timeout_val: int, is_greeting: bool = False, history: list = None):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    max_tokens = 30 if is_greeting else 1024
    contents = []
    if history:
        for msg in history:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    response = requests.post(
        url,
        json={
            "system_instruction": {"parts": {"text": "Отвечай на русском языке. Не используй Markdown. Отвечай кратко."}},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens}
        },
        timeout=timeout_val
    )
    
    if response.status_code == 503:
        raise Exception("Сервера Google сейчас перегружены. Попробуйте чуть позже.")
        
    data = response.json()
    if "error" in data:
        raise Exception("Ошибка Google API.")
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        raise Exception("Не удалось получить ответ от Google.")

def send_github_request(user_text: str, timeout_val: int, history: list = None):
    if not GITHUB_API_KEY: raise Exception("Не задан GITHUB_API_KEY.")
    messages = [{"role": "system", "content": "Отвечай на русском языке. Не используй Markdown. Отвечай кратко."}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    
    response = requests.post(
        GITHUB_API_URL,
        headers={"Authorization": f"Bearer {GITHUB_API_KEY}", "Content-Type": "application/json", "Accept": "application/vnd.github+json"},
        json={"model": "deepseek/DeepSeek-R1", "messages": messages},
        timeout=timeout_val
    )
    data = response.json()
    if "choices" not in data: raise Exception("Ошибка GitHub.")
    message = data["choices"][0]["message"]
    return (message.get("content") or message.get("reasoning_content") or "").strip()

def send_request_to_provider(model_key: str, user_text: str, timeout_val: int = 30, is_greeting: bool = False, user_id: str = None):
    history = get_history(user_id) if user_id else []
    if model_key == "джипити": return send_groq_request("openai/gpt-oss-120b", user_text, timeout_val, history)
    if model_key == "джемини": return send_gemini_request(user_text, timeout_val, is_greeting, history)
    if model_key == "дипсик": return send_github_request(user_text, timeout_val, history)
    raise Exception("Неизвестная модель.")

# --- ОБРАБОТКА WEBHOOK С НОРМАЛЬНЫМ ТАЙМАУТОМ ---

def do_sync_request_with_retry(func, *args, **kwargs):
    last_error = None
    start_time = time.time()
    
    for attempt in range(3):
        # Если уже прошло 3.8 секунды, нет смысла начинать новую попытку, Алиса вот-вот оборвет
        if time.time() - start_time >= 3.8:
            break
            
        try:
            # НОРМАЛЬНЫЙ ТАЙМАУТ 3.5 СЕКУНДЫ НА ПОПЫТКУ
            return func(*args, timeout_val=3.5, **kwargs)
        except requests.exceptions.RequestException as e:
            # Ловим Webhook (таймаут) или сетевую ошибку
            last_error = e
            time.sleep(0.1)
        except Exception as e:
            if "перегружены" in str(e):
                last_error = e
                time.sleep(0.5)
            else:
                raise e
                
    raise last_error

def warmup_model(model_key: str):
    """Синхронный прогрев. Ошибки игнорируются."""
    try:
        if model_key == "джипити": send_groq_request("openai/gpt-oss-120b", "hello", timeout_val=2.0)
        elif model_key == "джемини": send_gemini_request("hello", timeout_val=2.0, is_greeting=True)
        elif model_key == "дипсик": send_github_request("hello", timeout_val=2.0)
    except Exception:
        pass

def fetch_background(user_id: str, session_id: str, user_text: str, model_key: str):
    for attempt in range(3):
        try:
            answer = send_request_to_provider(model_key, user_text, timeout_val=30, user_id=user_id)
            CACHE[session_id] = split_text(answer, 900)
            update_history(user_id, "assistant", answer)
            break
        except Exception as e:
            if attempt == 2:
                if "перегружены" in str(e):
                    CACHE[session_id] = ["Сервера Google сейчас перегружены. Попробуйте чуть позже."]
                else:
                    CACHE[session_id] = [f"Ошибка после 3 попыток: {str(e)[:100]}"]
            else:
                time.sleep(1)

def get_intro_prompt(model_key: str):
    if model_key == "джипити": return "Поздоровайся с Сергеем по имени, как старый друг. Скажи что ты Джипити (GPT-OSS 120B)."
    if model_key == "джемини": return "Поздоровайся с Сергеем по имени, как старый друг. Скажи что ты Джемини (Gemini Flash от Google)."
    if model_key == "дипсик": return "Поздоровайся с Сергеем по имени, как старый друг. Скажи что ты Дипсик (DeepSeek)."

# --- ГЛАВНЫЙ ЭНДПОИНТ ---

@app.post("/")
async def main(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    req_data = body.get("request", {})
    user_text = (req_data.get("original_utterance") or req_data.get("command") or "").lower().strip()
    session_id = body["session"]["session_id"]
    user_id = body["session"]["user"]["user_id"]
    is_new_session = body.get("session", {}).get("new", False)

    current_model_key = USER_MODELS.get(user_id, "джипити")

    # 1. Меню
    if is_new_session or user_text in ["меню", "модели", "какая модель"]:
        menu_text = "Кибер Ноотроп запущен. Подключенные модели: Джипити — GPT-OSS 120B. Джемини — Gemini Flash. Дипсик — DeepSeek. Вернуть Алису. Какую модель подключить?"
        return {"version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": menu_text, "tts": menu_text}}

    # 2. Сброс
    if user_text in ["вернуть алису", "вернуться к алисе", "сбросить модель", "отключить модель", "по умолчанию"]:
        if user_id in USER_MODELS: del USER_MODELS[user_id]
        if user_id in HISTORY: del HISTORY[user_id]
        return {"version": body["version"], "session": body["session"],
                "response": {"end_session": True, "text": "Возврат к стандартной Алисе.", "tts": "Возврат к стандартной Алисе."}}

    # 3. Выбор модели
    found_model_key = None
    model_aliases = {
        "джипити": ["джипити", "gpt", "гпт", "джи пи ти"],
        "джемини": ["джемини", "gemini", "джемени"],
        "дипсик": ["дипсик", "deepseek", "дипсиг", "дип сик"]
    }
    clean_text = user_text.strip(".,!? ")
    for model_key, aliases in model_aliases.items():
        if clean_text in aliases:
            found_model_key = model_key
            break

    if found_model_key:
        if current_model_key == found_model_key:
            if session_id in CACHE and CACHE[session_id] != "PENDING":
                chunks = CACHE[session_id]
                if chunks:
                    current_chunk = chunks.pop(0)
                    if not chunks: del CACHE[session_id]
                    answer = current_chunk
                else:
                    answer = f"Модель {found_model_key.capitalize()} уже активна."
            elif session_id in CACHE and CACHE[session_id] == "PENDING":
                answer = "Приветствие еще генерируется, скажи «готово»."
            else:
                answer = f"Модель {found_model_key.capitalize()} уже активна."
            return {"version": body["version"], "session": body["session"],
                    "response": {"end_session": False, "text": answer, "tts": answer}}

        USER_MODELS[user_id] = found_model_key
        if session_id in CACHE: del CACHE[session_id]
        if user_id in HISTORY: del HISTORY[user_id]

        # Синхронный прогрев (ошибки игнорируются)
        warmup_model(found_model_key)

        intro_prompt = get_intro_prompt(found_model_key)
        CACHE[session_id] = "PENDING"

        try:
            # Запрос на приветствие с 3 попытками (внутри функция do_sync_request_with_retry)
            # Внешний wait_for дает 4.0 секунды, чтобы Алиса не оборвала связь
            loop = asyncio.get_event_loop()
            greeting = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: do_sync_request_with_retry(
                    send_request_to_provider, found_model_key, intro_prompt,
                    is_greeting=True, user_id=user_id
                )),
                timeout=4.0
            )
            answer = f"{greeting} Чтобы вернуться к списку моделей, скажите «меню». Чтобы вернуться к Алисе, скажите «вернуть алису»."
            del CACHE[session_id]
            return {"version": body["version"], "session": body["session"],
                    "response": {"end_session": False, "text": answer, "tts": answer}}
        except Exception as e:
            if "перегружены" in str(e):
                del CACHE[session_id]
                return {"version": body["version"], "session": body["session"],
                        "response": {"end_session": False, "text": str(e), "tts": str(e)}}

            # Если 3 попытки не удались или вышли за 4.0 сек, уходим в фон
            background_tasks.add_task(fetch_background, user_id, session_id, intro_prompt, found_model_key)
            answer = f"Подключаю модель {found_model_key.capitalize()}. Она формирует приветствие. Скажите «готово», чтобы услышать его."
            return {"version": body["version"], "session": body["session"],
                    "response": {"end_session": False, "text": answer, "tts": answer}}

    # 4. «дальше»
    if user_text in ["дальше", "дальше?"]:
        if session_id in CACHE and CACHE[session_id] != "PENDING":
            chunks = CACHE[session_id]
            if chunks:
                current_chunk = chunks.pop(0)
                if not chunks: del CACHE[session_id]
                else: current_chunk += "\n\nСкажи «дальше», чтобы послушать продолжение."
                return {"version": body["version"], "session": body["session"],
                        "response": {"end_session": False, "text": current_chunk, "tts": current_chunk}}
        answer = "Нет сохраненного продолжения. Задай новый вопрос."
        return {"version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": answer, "tts": answer}}

    # 5. «готово»
    if user_text in ["готово", "готово?"]:
        if session_id in CACHE:
            status = CACHE[session_id]
            if status == "PENDING":
                answer = "Еще обрабатываю, подожди 3-4 секунды и снова скажи «Готово»."
            else:
                chunks = status
                if chunks:
                    current_chunk = chunks.pop(0)
                    if not chunks: del CACHE[session_id]
                    else: current_chunk += "\n\nСкажи «дальше», чтобы послушать продолжение."
                    answer = current_chunk
                else:
                    answer = "Кэш пуст."
                    del CACHE[session_id]
        else:
            answer = "Нет активных фоновых задач."
        return {"version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": answer, "tts": answer}}

    # 6. Новый запрос
    if session_id in CACHE: del CACHE[session_id]

    update_history(user_id, "user", user_text)

    try:
        loop = asyncio.get_event_loop()
        answer = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: do_sync_request_with_retry(
                send_request_to_provider, current_model_key, user_text,
                is_greeting=False, user_id=user_id
            )),
            timeout=4.0
        )
        update_history(user_id, "assistant", answer)

        if len(answer) > 900:
            chunks = split_text(answer, 900)
            first_chunk = chunks.pop(0)
            CACHE[session_id] = chunks
            first_chunk += "\n\nОтвет длинный, скажи «дальше», чтобы продолжить."
            answer = first_chunk

        return {"version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": answer, "tts": answer}}

    except Exception as e:
        if "перегружены" in str(e):
            return {"version": body["version"], "session": body["session"],
                    "response": {"end_session": False, "text": str(e), "tts": str(e)}}

        # Уходим в фон
        CACHE[session_id] = "PENDING"
        background_tasks.add_task(fetch_background, user_id, session_id, user_text, current_model_key)
        answer = "Запрос сложный, обрабатываю в фоне. Подожди 5 секунд и спроси «Готово?»"
        return {"version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": answer, "tts": answer}}
