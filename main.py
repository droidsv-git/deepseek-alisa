import os
import asyncio
import requests
import time
import psutil  # ДОБАВЛЕНО ДЛЯ КОМАНДЫ РЕСУРСОВ
from fastapi import FastAPI, Request, BackgroundTasks

app = FastAPI()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GPT_API_KEY = os.getenv("GPT_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

RENDER_URL = "https://deepseek-alisa-fusq.onrender.com"

CACHE = {}
USER_MODELS = {}
HISTORY = {}

SESSION_TTL = 1800  # 30 минут

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
    return (message.get("content") or message.get("reasoning") or "").strip()

def send_gemini_request(user_text: str, timeout_val: int, is_greeting: bool = False, history: list = None):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    max_tokens = 30 if is_greeting else 1024
    
    trimmed_history = history[-6:] if history else []
    
    contents = []
    if trimmed_history:
        for msg in trimmed_history:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
    
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    for attempt in range(3):
        try:
            response = requests.post(
                url,
                json={
                    "system_instruction": {"parts": {"text": "Отвечай на русском языке. Не используй Markdown. Отвечай кратко."}},
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": max_tokens
                    }
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
                    raise Exception("Ошибка Google API.")
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                raise Exception("Не удалось получить ответ от Google.")
        except Exception as e:
            if str(e) != "503" or attempt == 2:
                if str(e) == "503":
                    raise Exception("Сервера Google сейчас перегружены. Попробуйте чуть позже.")
                raise e
            time.sleep(1 * (2 ** attempt))

def send_openrouter_request(user_text: str, timeout_val: int, history: list = None):
    if not OPENROUTER_API_KEY:
        raise Exception("Не задан OPENROUTER_API_KEY.")
    messages = [{"role": "system", "content": "Отвечай на русском языке. Не используй Markdown. Отвечай кратко."}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    
    response = requests.post(
        OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://deepseek-alisa-fusq.onrender.com",
            "X-Title": "Alisa Skill"
        },
        json={
            "model": "openrouter/free",
            "messages": messages
        },
        timeout=timeout_val
    )
    data = response.json()
    if "choices" not in data:
        raise Exception(f"Ошибка OpenRouter: {data}")
    message = data["choices"][0]["message"]
    return (message.get("content") or message.get("reasoning_content") or "").strip()

def send_request_to_provider(model_key: str, user_text: str, timeout_val: int = 30, is_greeting: bool = False, user_id: str = None):
    history = get_history(user_id) if user_id else []
    if model_key == "джипити":
        return send_groq_request("openai/gpt-oss-120b", user_text, timeout_val, history)
    if model_key == "джемини":
        return send_gemini_request(user_text, timeout_val, is_greeting, history)
    if model_key == "опенроутер":
        return send_openrouter_request(user_text, timeout_val, None) 
    raise Exception("Неизвестная модель.")

def get_intro_prompt(model_key: str):
    if model_key == "джипити":
        return "Поздоровайся с Сергеем по имени, как старый друг. Скажи что ты Джипити (GPT-OSS 120B)."
    if model_key == "джемини":
        return "Поздоровайся с Сергеем по имени, как старый друг. Скажи что ты Джемини (Gemini Flash от Google)."
    if model_key == "опенроутер":
        return "Поздоровайся с Сергеем по имени, как старый друг. Скажи что ты Опен Роутер (OpenRouter)."

def fetch_background(user_id: str, session_id: str, user_text: str, model_key: str):
    try:
        answer = send_request_to_provider(model_key, user_text, timeout_val=30, user_id=user_id)
        CACHE[session_id] = split_text(answer, 900)
        update_history(user_id, "assistant", answer)
    except Exception as e:
        if "перегружены" in str(e):
            CACHE[session_id] = ["Сервера Google сейчас перегружены. Попробуйте чуть позже."]
        else:
            CACHE[session_id] = [f"Ошибка: {str(e)[:150]}"]

async def safe_sync_call(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(loop.run_in_executor(None, lambda: func(*args, **kwargs)), timeout=4.0)

@app.post("/")
async def main(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    req_data = body.get("request", {})
    user_text = (req_data.get("original_utterance") or req_data.get("command") or "").lower().strip()
    session_id = body["session"]["session_id"]
    user_id = body["session"]["user"]["user_id"]
    is_new_session = body.get("session", {}).get("new", False)

    if is_new_session:
        USER_MODELS.pop(user_id, None)
        HISTORY.pop(user_id, None)

    current_model_key = USER_MODELS.get(user_id, "")

    # --- СКРЫТАЯ КОМАНДА РЕСУРСОВ ---
    clean_cmd = user_text.strip(".,!? ")
    clean_cmd_no_spaces = "".join(clean_cmd.split())
    
    resource_commands = [
        "ресурсы сервера ноотроп", "ресурсы сервера", "ресурсы ноотропа", "ресурсы ноотроп",
        "ресурсы сервер", "ресурс сервера", "ресурс сервер", "ресурсы кибер ноотроп",
        "покажи ресурсы", "загрузка сервера", "нагрузка сервера", "состояние сервера",
        "статус сервера", "ресурсы системы", "ресурсы ноутроп", "ресурсы неотрон", 
        "ресурсы нотрон", "сервер ресурсы", "сервер загрузка", "ресурсы сервера ноутроп",
        "ресурсы кибер", "кибер ресурсы", "ресурсы сервера нооттроп", "ресурсы сервер ноотроп",
        "ресурсы ноутроп", "ресурсы неотроп", "ресурсы нотроп", "ресурсы ноотроп", "ресурсы сервера неотроп",
        "мониторинг сервера", "мониторинг", "нагрузка", "ресурсы"
    ]
    resource_commands_no_spaces = [cmd.replace(" ", "") for cmd in resource_commands]
    
    if clean_cmd_no_spaces in resource_commands_no_spaces:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            total_ram_gb = round(mem.total / (1024**3), 1)
            used_ram_percent = mem.percent
            
            net1 = psutil.net_io_counters()
            time.sleep(0.2)
            net2 = psutil.net_io_counters()
            
            sent_kbs = round(((net2.bytes_sent - net1.bytes_sent) / 0.2) / 1024, 1)
            recv_kbs = round(((net2.bytes_recv - net1.bytes_recv) / 0.2) / 1024, 1)

            cpu_model = "Неизвестный процессор"
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_model = line.split(":")[1].strip()
                            break
            except Exception:
                pass
            
            text_answer = (
                f"Ресурсы вашего сервера кибер ноотроп. "
                f"Процессор: {cpu_model}, загружен на {int(cpu_percent)} процентов. "
                f"Оперативная память: {total_ram_gb} гигабайт, использовано {int(used_ram_percent)} процентов. "
                f"Сетевой трафик: отдача {int(sent_kbs)} килобайт в секунду, прием {int(recv_kbs)} килобайт в секунду."
            )
            
            tts_answer = (
                f"Ресурсы вашего сервера кибер ноотроп. sil <[600]> "
                f"Процессор: {cpu_model}, загружен на {int(cpu_percent)} процентов. "
                f"Оперативная память: {total_ram_gb} гигабайт, использовано {int(used_ram_percent)} процентов. "
                f"Сетевой трафик: отдача {int(sent_kbs)} килобайт в секунду, прием {int(recv_kbs)} килобайт в секунду."
            )
        except Exception:
            text_answer = "Ресурсы вашего сервера кибер ноотроп. Не удалось получить данные о системе."
            tts_answer = text_answer
            
        return {"version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": text_answer, "tts": tts_answer}}
    # ---------------------------------

    if is_new_session or user_text in ["меню", "модели", "какая модель"]:
        menu_text = (
            "Кибер Ноотроп запущен. Подключенные модели: "
            "Джипити — GPT-OSS 120B. Джемини — Gemini Flash. Опен Роутер — OpenRouter Free. "
            "Вернуть Алису. Какую модель подключить?"
        )
        return {"version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": menu_text, "tts": menu_text}}

    # --- РАСШИРЕННЫЙ СПИСОК "ВЕРНУТЬ АЛИСУ" ---
    clean_cmd = user_text.strip(".,!? ")
    clean_cmd_no_spaces = "".join(clean_cmd.split())
    
    alisa_commands = [
        "вернуть алису", "вернуться к алисе", "сбросить модель", "отключить модель", "по умолчанию",
        "верни алису", "верни алиса", "вернись алиса", "верну алису", "вернуть алиса",
        "вернись к алисе", "отключи модель", "сбрось модель", "сброс модели", "алису верни",
        "алису вернуть", "назад к алисе", "стандартная алиса", "без модели", "отключить алису",
        "отключи алису", "завершить", "выйти", "хватит"
    ]
    alisa_commands_no_spaces = [cmd.replace(" ", "") for cmd in alisa_commands]
    
    if clean_cmd_no_spaces in alisa_commands_no_spaces:
        if user_id in USER_MODELS: del USER_MODELS[user_id]
        if user_id in HISTORY: del HISTORY[user_id]
        return {"version": body["version"], "session": body["session"],
                "response": {"end_session": True, "text": "Возврат к стандартной Алисе.", "tts": "Возврат к стандартной Алисе."}}
    # ---------------------------------------------------

    found_model_key = None
    # --- РАСШИРЕННЫЙ СПИСОК ОПЕНРОУТЕРА ---
    model_aliases = {
        "джипити": ["джипити", "gpt", "гпт", "джи пи ти"],
        "джемини": ["джемини", "gemini", "джемени"],
        "опенроутер": [
            "опен роутер", "опенроутер", "openrouter", "open router", 
            "опэн роутер", "опэнроутер", "оупен роутер", "оупенроутер",
            "опен рутер", "опенрутер", "опен раутер", "опенраутер",
            "опен рутор", "опенрутор", "апи роутер", "апин роутер",
            "опен раутор", "опенраутор", "опен раутэр", "роутер",
            "опен рутэр", "опэн рутер", "оупэн роутер", "опен рудэр",
            "опен рудер", "опенрудер", "оупен рутер", "опэн раутер",
            "опен сервер", "опенсервер", "опен сэр вер", "опэн сервер",
            "оупен сервер", "апи сервер", "апин сервер", "опен сервэр",
            "опен сивер", "опен север", "опенсервис", "опен сервис", 
            "апин сэрвер", "опенсэвэр", "опенсевер", "агрегатор", "шлюз",
            "нейро сеть", "нейросеть", "open", "опен"
        ]
    }
    # ---------------------------------------------------
    
    clean_text = user_text.strip(".,!? ")
    clean_text_no_spaces = "".join(clean_text.split())

    for model_key, aliases in model_aliases.items():
        for alias in aliases:
            alias_clean_no_spaces = "".join(alias.replace(" ", "").split())
            if clean_text_no_spaces == alias_clean_no_spaces:
                found_model_key = model_key
                break
        if found_model_key:
            break

    if found_model_key:
        if current_model_key == found_model_key:
            answer = f"Модель {found_model_key.capitalize()} уже активна."
            return {"version": body["version"], "session": body["session"],
                    "response": {"end_session": False, "text": answer, "tts": answer}}

        USER_MODELS[user_id] = found_model_key
        if session_id in CACHE: del CACHE[session_id]
        if user_id in HISTORY: del HISTORY[user_id]

        intro_prompt = get_intro_prompt(found_model_key)
        try:
            greeting = await safe_sync_call(
                send_request_to_provider, found_model_key, intro_prompt,
                timeout_val=4.0, is_greeting=True, user_id=user_id
            )
            answer = f"{greeting} Чтобы вернуться к списку моделей, скажите «меню». Чтобы вернуться к Алисе, скажите «вернуть алису»."
            return {"version": body["version"], "session": body["session"],
                    "response": {"end_session": False, "text": answer, "tts": answer}}
        except Exception as e:
            if "перегружены" in str(e):
                return {"version": body["version"], "session": body["session"],
                        "response": {"end_session": False, "text": str(e), "tts": str(e)}}
            CACHE[session_id] = "PENDING"
            background_tasks.add_task(fetch_background, user_id, session_id, intro_prompt, found_model_key)
            answer = f"Подключаю модель {found_model_key.capitalize()}. Она формирует приветствие. Скажите «готово», чтобы услышать его."
            return {"version": body["version"], "session": body["session"],
                    "response": {"end_session": False, "text": answer, "tts": answer}}

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

    if session_id in CACHE: del CACHE[session_id]

    update_history(user_id, "user", user_text)

    try:
        answer = await safe_sync_call(
            send_request_to_provider, current_model_key, user_text,
            timeout_val=4.0, is_greeting=False, user_id=user_id
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

        CACHE[session_id] = "PENDING"
        background_tasks.add_task(fetch_background, user_id, session_id, user_text, current_model_key)
        answer = "Запрос сложный, обрабатываю в фоне. Подожди 5 секунд и спроси «Готово?»"
        return {"version": body["version"], "session": body["session"],
                "response": {"end_session": False, "text": answer, "tts": answer}}
