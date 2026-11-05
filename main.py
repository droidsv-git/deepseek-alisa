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

# --- КЭШ И ПАМЯТЬ ---
CACHE = {}          # Для длинных ответов (нарезка по 900 символов)
USER_MODELS = {}    # Активная модель для сессии
HISTORY = {}        # История диалога: {session_id: {"data": [...], "last_update": timestamp}}

# Время жизни сессии (30 минут = 1800 секунд)
SESSION_TTL = 1800 

def get_history(session_id: str):
    """Возвращает историю, если сессия не устарела, иначе сбрасывает её."""
    if session_id in HISTORY:
        # Проверяем, не устарела ли сессия
        if time.time() - HISTORY[session_id]["last_update"] < SESSION_TTL:
            return HISTORY[session_id]["data"]
        else:
            # Сессия устарела, удаляем
            del HISTORY[session_id]
    return []

def update_history(session_id: str, role: str, content: str):
    """Обновляет историю и время последнего обращения."""
    if session_id not in HISTORY:
        HISTORY[session_id] = {"data": [], "last_update": time.time()}
    
    HISTORY[session_id]["data"].append({"role": role, "content": content})
    HISTORY[session_id]["last_update"] = time.time()
    
    # Ограничиваем историю 20 сообщениями (10 пар), чтобы не переполнять контекст
    if len(HISTORY[session_id]["data"]) > 20:
        HISTORY[session_id]["data"] = HISTORY[session_id]["data"][-20:]

# --- БАЗОВЫЕ ФУНКЦИИ ---

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

# --- ЗАПРОСЫ К API С RETRY ДЛЯ 503 ---

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
    if "choices" not in data:
        raise Exception("Ошибка Groq.")
    message = data["choices"][0]["message"]
    answer = message.get("content") or message.get("reasoning") or message.get("reasoning_content") or ""
    return answer.strip()

def send_gemini_request(user_text: str, timeout_val: int, is_greeting: bool = False, history: list = None):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    
    max_tokens = 30 if is_greeting else 1024
    contents = []
    
    if history:
        for msg in history:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})
    
    # Экспоненциальная задержка для 503
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
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
                raise Exception("503")
            
            data = response.json()
            
            if "error" in data:
                if data["error"].get("code") == 503:
                    raise Exception("503")
                else:
                    raise Exception(f"Ошибка Google API: {data['error'].get('message', 'Неизвестная ошибка')}")
                    
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                raise Exception("Не удалось получить ответ от Google.")
                
        except Exception as e:
            # Если это не 503 или это последняя попытка, пробрасываем ошибку дальше
            if str(e) != "503" or attempt == max_retries - 1:
                if str(e) == "503":
                    raise Exception("Сервера Google перегружены. Попробуйте позже.")
                raise e
            
            # Ждем перед следующей попыткой (1с, 2с, 4с)
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)

def send_github_request(user_text: str, timeout_val: int, history: list = None):
    if not GITHUB_API_KEY:
        raise Exception("Не задан GITHUB_API_KEY.")
    
    messages = [{"role": "system", "content": "Отвечай на русском языке. Не используй Markdown. Отвечай кратко."}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    
    response = requests.post(
        GITHUB_API_URL,
        headers={
            "Authorization": f"Bearer {GITHUB_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json"
        },
        json={"model": "deepseek/DeepSeek-R1", "messages": messages},
        timeout=timeout_val
    )
    data = response.json()
    if "choices" not in data:
        raise Exception("Ошибка GitHub.")
    message = data["choices"][0]["message"]
    answer = message.get("content") or message.get("reasoning_content") or ""
    return answer.strip()

def send_request_to_provider(model_key: str, user_text: str, timeout_val: int = 30, is_greeting: bool = False, session_id: str = None):
    history = get_history(session_id) if session_id else []
    
    if model_key == "джипити":
        return send_groq_request("openai/gpt-oss-120b", user_text, timeout_val, history)
    if model_key == "джемини":
        return send_gemini_request(user_text, timeout_val, is_greeting, history)
    if model_key == "дипсик":
        return send_github_request(user_text, timeout_val, history)
    raise Exception("Неизвестная модель.")

# --- ЛОГИКА ПРИВЕТСТВИЯ И ФОНОВЫХ ЗАДАЧ ---

def get_intro_prompt(model_key: str):
    if model_key == "джипити":
        return "Поздоровайся с Сергеем по имени, как старый друг. Скажи что ты Джипити (GPT-OSS 120B)."
    if model_key == "джемини":
        return "Поздоровайся с Сергеем по имени, как старый друг. Скажи что ты Джемини (Gemini Flash от Google)."
    if model_key == "дипсик":
        return "Поздоровайся с Сергеем по имени, как старый друг. Скажи что ты Дипсик (DeepSeek)."

def fetch_background(session_id: str, user_text: str, model_key: str):
    try:
        answer = send_request_to_provider(model_key, user_text, timeout_val=30, is_greeting=False, session_id=session_id)
        chunks = split_text(answer, 900)
        CACHE[session_id] = chunks
        
        # Сохраняем в долгую память
        update_history(session_id, "user", user_text)
        update_history(session_id, "assistant", answer)
        
    except Exception as e:
        # Если в фоне произошла ошибка 503, показываем понятный текст
        if "перегружены" in str(e):
            CACHE[session_id] = ["Сервера Google сейчас перегружены. Попробуйте задать вопрос чуть позже."]
        else:
            CACHE[session_id] = [f"Ошибка: {str(e)[:150]}"]

# --- ГЛАВНЫЙ ЭНДПОИНТ ---

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
        if session_id in USER_MODELS: del USER_MODELS[session_id]
        if session_id in HISTORY: del HISTORY[session_id]
        return {
            "version": body["version"], "session": body["session"],
            "response": {"end_session": True, "text": "Возврат к стандартной Алисе.", "tts": "Возврат к стандартной Алисе."}
        }

    # 3. Выбор модели (строгое совпадение)
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
        # Если модель уже активна, не здороваемся заново
        if current_model_key == found_model_key:
            answer = f"Модель {found_model_key.capitalize()} уже активна."
            return {
                "version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": answer, "tts": answer}
            }
            
        USER_MODELS[session_id] = found_model_key
        if session_id in CACHE: del CACHE[session_id]
        # Сбрасываем историю при смене модели, чтобы не путать контексты
        if session_id in HISTORY: del HISTORY[session_id]

        intro_prompt = get_intro_prompt(found_model_key)
        try:
            # Таймаут 4.5 секунды — максимальный запас для Алисы
            greeting = send_request_to_provider(found_model_key, intro_prompt, timeout_val=4.5, is_greeting=True, session_id=session_id)
            answer = f"{greeting} Чтобы вернуться к списку моделей, скажите «меню». Чтобы вернуться к Алисе, скажите «вернуть алису»."
            return {
                "version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": answer, "tts": answer}
            }
        except Exception as e:
            if "перегружены" in str(e):
                return {
                    "version": body["version"], "session": body["session"],
                    "response": {"end_session": False, "text": str(e), "tts": str(e)}
                }
            CACHE[session_id] = "PENDING"
            background_tasks.add_task(fetch_background, session_id, intro_prompt, found_model_key)
            answer = f"Подключаю модель {found_model_key.capitalize()}. Она формирует приветствие. Скажите «готово», чтобы услышать его."
            return {
                "version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": answer, "tts": answer}
            }

    # 4. «дальше»
    if user_text in ["дальше", "дальше?"]:
        if session_id in CACHE and CACHE[session_id] != "PENDING":
            chunks = CACHE[session_id]
            if chunks:
                current_chunk = chunks.pop(0)
                if not chunks: del CACHE[session_id]
                else: current_chunk += "\n\nСкажи «дальше», чтобы послушать продолжение."
                return {
                    "version": body["version"], "session": body["session"],
                    "response": {"end_session": False, "text": current_chunk, "tts": current_chunk}
                }
        answer = "Нет сохраненного продолжения. Задай новый вопрос."
        return {
            "version": body["version"], "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }

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
        return {
            "version": body["version"], "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }

    # 6. Новый запрос
    if session_id in CACHE: del CACHE[session_id]

    try:
        # Таймаут 4.0 секунды для синхронного ответа
        answer = send_request_to_provider(current_model_key, user_text, timeout_val=4.0, is_greeting=False, session_id=session_id)

        # Сохраняем в память
        update_history(session_id, "user", user_text)
        update_history(session_id, "assistant", answer)

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

    except Exception as e:
        if "перегружены" in str(e):
            return {
                "version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": str(e), "tts": str(e)}
            }
            
        CACHE[session_id] = "PENDING"
        background_tasks.add_task(fetch_background, session_id, user_text, current_model_key)
        answer = "Запрос сложный, обрабатываю в фоне. Подожди 5 секунд и спроси «Готово?»"
        return {
            "version": body["version"], "session": body["session"],
            "response": {"end_session": False, "text": answer, "tts": answer}
        }
