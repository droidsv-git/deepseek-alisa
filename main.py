import os
import asyncio
import requests
from fastapi import FastAPI, Request, BackgroundTasks

app = FastAPI()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GPT_API_KEY = os.getenv("GPT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

RENDER_URL = "https://deepseek-alisa-fusq.onrender.com"

CACHE = {}
USER_MODELS = {}

# АКТУАЛЬНЫЕ бесплатные модели:
# GPT — Groq, openai/gpt-oss-120b
# Gemini — Google, gemini-2.5-flash
# DeepSeek — Groq, deepseek-r1-distill-llama-70b
AVAILABLE_MODELS = {
    "джипити": {
        "aliases": ["джипити", "gpt", "гпт", "джи пи ти"],
        "provider": "groq",
        "path": "openai/gpt-oss-120b",
        "intro_prompt": (
            "Представься от первого лица на русском языке. "
            "Поприветствуй пользователя, скажи, что тебя зовут Джипити "
            "и что ты языковая модель GPT-OSS 120B. "
            "Ответь коротко, вежливо и без Markdown."
        )
    },
    "джемини": {
        "aliases": ["джемини", "gemini", "джемени"],
        "provider": "gemini",
        "path": "gemini-2.5-flash",
        "intro_prompt": (
            "Представься от первого лица на русском языке. "
            "Поприветствуй пользователя, скажи, что тебя зовут Джемини "
            "и что ты языковая модель Gemini Flash от Google. "
            "Ответь коротко, вежливо и без Markdown."
        )
    },
    "дипсик": {
        "aliases": ["дипсик", "deepseek", "дипсиг", "дип сик"],
        "provider": "groq",
        "path": "deepseek-r1-distill-llama-70b",
        "intro_prompt": (
            "Представься от первого лица на русском языке. "
            "Поприветствуй пользователя, скажи, что тебя зовут Дипсик "
            "и что ты языковая модель DeepSeek R1. "
            "Ответь коротко, вежливо и без Markdown."
        )
    }
}


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


def send_request_to_provider(model_key: str, user_text: str, timeout_val: int = 30):
    m_info = AVAILABLE_MODELS[model_key]
    provider = m_info["provider"]
    model_path = m_info["path"]

    if provider == "groq":
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

        # Для DeepSeek R1 иногда основной текст находится в reasoning
        answer = (
            message.get("content")
            or message.get("reasoning")
            or message.get("reasoning_content")
            or ""
        )

        return answer.strip()

    elif provider == "gemini":
        gemini_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_path}:generateContent?key={GEMINI_API_KEY}"
        )

        response = requests.post(
            gemini_url,
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


def fetch_greeting(session_id: str, model_key: str):
    try:
        intro_prompt = AVAILABLE_MODELS[model_key]["intro_prompt"]

        # Приветствие генерирует САМА выбранная модель через её API
        answer = send_request_to_provider(
            model_key,
            intro_prompt,
            timeout_val=45
        )

        chunks = split_text(answer, 900)
        CACHE[session_id] = chunks

    except Exception as e:
        CACHE[session_id] = [
            f"Не удалось получить приветствие: {str(e)[:150]}"
        ]


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
            "Дипсик — DeepSeek R1. "
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

    for m_key, m_data in AVAILABLE_MODELS.items():
        for alias in m_data["aliases"]:
            if alias in user_text:
                found_model_key = m_key
                break

        if found_model_key:
            break

    if found_model_key:
        USER_MODELS[session_id] = found_model_key

        # Сбрасываем предыдущий ответ
        if session_id in CACHE:
            del CACHE[session_id]

        # Выбранная модель сама генерирует приветствие через API
        CACHE[session_id] = "PENDING"

        background_tasks.add_task(
            fetch_greeting,
            session_id,
            found_model_key
        )

        answer = (
            f"Подключаю модель {found_model_key.capitalize()}. "
            "Чтобы услышать её приветствие, скажите «готово»."
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
        # ВАШ таймаут не изменён: 3.5 секунды
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
