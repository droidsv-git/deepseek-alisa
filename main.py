import os
import asyncio
import requests
from fastapi import FastAPI, Request, BackgroundTasks

app = FastAPI()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GPT_API_KEY = os.getenv("GPT_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"

SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
SAMBANOVA_API_URL = "https://api.sambanova.ai/v1/chat/completions"
SAMBANOVA_MODELS_URL = "https://api.sambanova.ai/v1/models"

DEEPSEEK_MODEL = None

RENDER_URL = "https://deepseek-alisa-fusq.onrender.com"

CACHE = {}
USER_MODELS = {}


def load_deepseek_model():
    global DEEPSEEK_MODEL

    try:
        response = requests.get(
            SAMBANOVA_MODELS_URL,
            headers={"Authorization": f"Bearer {SAMBANOVA_API_KEY}"},
            timeout=10
        )

        data = response.json()

        models = data.get("data", [])

        deepseek_models = [
            model["id"]
            for model in models
            if "deepseek" in model.get("id", "").lower()
        ]

        if deepseek_models:
            DEEPSEEK_MODEL = deepseek_models[0]

    except Exception:
        DEEPSEEK_MODEL = None


@app.on_event("startup")
async def startup_event():
    load_deepseek_model()

    async def self_ping():
        while True:
            await asyncio.sleep(600)
            try:
                requests.get(RENDER_URL, timeout=5)
            except Exception:
                pass

    asyncio.create_task(self_ping())


@app.get("/")
async def root():
    return {"status": "awake"}


def get_deepseek_model():
    global DEEPSEEK_MODEL

    if not DEEPSEEK_MODEL:
        load_deepseek_model()

    return DEEPSEEK_MODEL


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


def send_groq_request(model_path: str, user_text: str, timeout_val: int):
    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {GPT_API_KEY}"},
        json={
            "model": model_path,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Отвечай на русском языке. "
                        "Не используй Markdown. "
                        "Не показывай внутренние размышления."
                    )
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ]
        },
        timeout=timeout_val
    )

    data = response.json()

    if "choices" not in data:
        raise Exception(f"Groq API Error: {data}")

    message = data["choices"][0]["message"]

    answer = (
        message.get("content")
        or message.get("reasoning")
        or message.get("reasoning_content")
        or ""
    )

    return answer.strip()


def send_gemini_request(user_text: str, timeout_val: int):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    response = requests.post(
        url,
        json={
            "contents": [
                {
                    "parts": [
                        {"text": user_text}
                    ]
                }
            ]
        },
        timeout=timeout_val
    )

    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        raise Exception(f"Gemini API Error: {data}")


def send_sambanova_request(user_text: str, timeout_val: int):
    if not SAMBANOVA_API_KEY:
        raise Exception("Не задан SAMBANOVA_API_KEY.")

    model_path = get_deepseek_model()

    if not model_path:
        raise Exception(
            "DeepSeek недоступен в SambaNova. "
            "Проверьте список моделей: https://api.sambanova.ai/v1/models"
        )

    response = requests.post(
        SAMBANOVA_API_URL,
        headers={
            "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": model_path,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Отвечай на русском языке. "
                        "Не используй Markdown. "
                        "Не показывай внутренние размышления."
                    )
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ]
        },
        timeout=timeout_val
    )

    data = response.json()

    if "choices" not in data:
        raise Exception(f"SambaNova DeepSeek API Error: {data}")

    message = data["choices"][0]["message"]

    answer = (
        message.get("content")
        or message.get("reasoning_content")
        or ""
    )

    return answer.strip()


def send_request_to_provider(model_key: str, user_text: str, timeout_val: int = 30):
    if model_key == "джипити":
        return send_groq_request(
            model_path="openai/gpt-oss-120b",
            user_text=user_text,
            timeout_val=timeout_val
        )

    if model_key == "джемини":
        return send_gemini_request(
            user_text=user_text,
            timeout_val=timeout_val
        )

    if model_key == "дипсик":
        return send_sambanova_request(
            user_text=user_text,
            timeout_val=timeout_val
        )

    raise Exception("Неизвестная модель.")


def get_intro_prompt(model_key: str):
    if model_key == "джипити":
        return (
            "Представься от первого лица на русском языке. "
            "Поприветствуй пользователя, скажи, что тебя зовут Джипити "
            "и что ты языковая модель GPT-OSS 120B. "
            "Ответь коротко, вежливо и без Markdown."
        )

    if model_key == "джемини":
        return (
            "Представься от первого лица на русском языке. "
            "Поприветствуй пользователя, скажи, что тебя зовут Джемини "
            "и что ты языковая модель Gemini Flash от Google. "
            "Ответь коротко, вежливо и без Markdown."
        )

    if model_key == "дипсик":
        return (
            "Представься от первого лица на русском языке. "
            "Поприветствуй пользователя, скажи, что тебя зовут Дипсик "
            "и что ты языковая модель DeepSeek. "
            "Ответь коротко, вежливо и без Markdown."
        )


def fetch_background(session_id: str, user_text: str, model_key: str):
    try:
        answer = send_request_to_provider(
            model_key,
            user_text,
            timeout_val=45
        )

        chunks = split_text(answer, 900)
        CACHE[session_id] = chunks

    except Exception as e:
        CACHE[session_id] = [f"Ошибка генерации: {str(e)[:150]}"]


@app.post("/")
async def main(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()

    req_data = body.get("request", {})
    user_text = (
        req_data.get("original_utterance")
        or req_data.get("command")
        or ""
    ).lower().strip()

    session_id = body["session"]["session_id"]
    is_new_session = body.get("session", {}).get("new", False)

    current_model_key = USER_MODELS.get(session_id, "джипити")

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
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": menu_text,
                "tts": menu_text
            }
        }

    if user_text in [
        "вернуть алису",
        "вернуться к алисе",
        "сбросить модель",
        "отключить модель",
        "по умолчанию"
    ]:
        if session_id in USER_MODELS:
            del USER_MODELS[session_id]

        answer = "Возврат к стандартной Алисе."

        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": True,
                "text": answer,
                "tts": answer
            }
        }

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
            # Приветствие генерирует выбранная модель через API
            # и сразу озвучивается Станцией.
            # Таймаут не изменён: 3.5 секунды.
            greeting = send_request_to_provider(
                found_model_key,
                intro_prompt,
                timeout_val=3.5
            )

            answer = (
                f"{greeting} "
                "Чтобы вернуться к списку моделей, скажите «меню». "
                "Чтобы вернуться к Алисе, скажите «вернуть алису»."
            )

            return {
                "version": body["version"],
                "session": body["session"],
                "response": {
                    "end_session": False,
                    "text": answer,
                    "tts": answer
                }
            }

        except Exception:
            CACHE[session_id] = "PENDING"

            background_tasks.add_task(
                fetch_background,
                session_id,
                intro_prompt,
                found_model_key
            )

            answer = (
                f"Подключаю модель {found_model_key.capitalize()}. "
                "Она формирует приветствие. "
                "Скажите «готово», чтобы услышать его."
            )

            return {
                "version": body["version"],
                "session": body["session"],
                "response": {
                    "end_session": False,
                    "text": answer,
                    "tts": answer
                }
            }

    if user_text in ["дальше", "дальше?"]:
        if session_id in CACHE and CACHE[session_id] != "PENDING":
            chunks = CACHE[session_id]

            if chunks:
                current_chunk = chunks.pop(0)

                if not chunks:
                    del CACHE[session_id]
                else:
                    current_chunk += (
                        "\n\nСкажи «дальше», чтобы послушать продолжение."
                    )

                return {
                    "version": body["version"],
                    "session": body["session"],
                    "response": {
                        "end_session": False,
                        "text": current_chunk,
                        "tts": current_chunk
                    }
                }

        answer = "Нет сохраненного продолжения. Задай новый вопрос."

        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": answer,
                "tts": answer
            }
        }

    if user_text in ["готово", "готово?"]:
        if session_id in CACHE:
            status = CACHE[session_id]

            if status == "PENDING":
                answer = (
                    "Еще думаю над ответом, "
                    "спроси «Готово?» через секунду."
                )
            else:
                chunks = status

                if chunks:
                    current_chunk = chunks.pop(0)

                    if not chunks:
                        del CACHE[session_id]
                    else:
                        current_chunk += (
                            "\n\nСкажи «дальше», "
                            "чтобы послушать продолжение."
                        )

                    answer = current_chunk
                else:
                    answer = "Кэш пуст."
                    del CACHE[session_id]
        else:
            answer = "Нет активных фоновых задач."

        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": answer,
                "tts": answer
            }
        }

    if session_id in CACHE:
        del CACHE[session_id]

    try:
        # Ваш таймаут не изменён: 3.5 секунды
        answer = send_request_to_provider(
            current_model_key,
            user_text,
            timeout_val=3.5
        )

        if len(answer) > 900:
            chunks = split_text(answer, 900)
            first_chunk = chunks.pop(0)
            CACHE[session_id] = chunks

            first_chunk += (
                "\n\nОтвет длинный, скажи «дальше», чтобы продолжить."
            )

            answer = first_chunk

        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": answer,
                "tts": answer
            }
        }

    except Exception:
        CACHE[session_id] = "PENDING"

        background_tasks.add_task(
            fetch_background,
            session_id,
            user_text,
            current_model_key
        )

        answer = (
            "Вопрос сложный, обдумываю в фоне. "
            "Спроси «Готово?» через пару секунд."
        )

        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": answer,
                "tts": answer
            }
        }
