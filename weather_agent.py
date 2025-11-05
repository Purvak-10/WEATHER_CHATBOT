import os
import json
import requests
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union, Literal

from pydantic import BaseModel, field_validator
# from langchain import LLMChain, PromptTemplate
# from langchain_community.llms import Ollama

from groq import Groq



# Load environment variables
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
# OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not OPENWEATHER_API_KEY:
    raise RuntimeError("OPENWEATHER_API_KEY not found in environment. Add it to .env")



class IntentCityModel(BaseModel):
    intent: Literal["get_weather", "other"]
    cities: List[str] = []

    # ✅ Normalize cities BEFORE validation
    @field_validator("cities", mode="before")
    def normalize_city_list(cls, value):
        if value is None:
            return []

        # Case 1: Already a list
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    cleaned.append(item.strip().title())
            return cleaned

        # Case 2: Single string from LLM
        if isinstance(value, str):
            # Split on comma or "and"
            parts = re.split(r",|\band\b", value)
            cleaned = [p.strip().title() for p in parts if p.strip()]
            return cleaned

        # Unknown type → return empty list
        return []

    # Normalize intent BEFORE validation
    @field_validator("intent", mode="before")
    def normalize_intent(cls, v):
        if not isinstance(v, str):
            return "other"
        return "get_weather" if v.strip().lower() == "get_weather" else "other"


class WeatherResult(BaseModel):
    city: str
    weather: str
    error: bool = False
    message: Optional[str] = None

    @field_validator("city", mode="before")
    def normalize_city(cls, v):
        if isinstance(v, str):
            return v.strip().title()
        return str(v)

    @field_validator("weather", mode="before")
    def normalize_weather(cls, v):
        if v is None:
            return "data not available"
        return str(v).strip()


# LLM Helper

# # ollama LLM instance
# def get_llm():
#     """Return a configured Ollama LLM instance."""
#     return Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_API_URL, verbose=False, timeout=120)

# groq LLM instance
def get_llm():
    """Return a callable that sends prompts to Groq's Llama 3.1 model."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def call(prompt: str) -> str:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        return completion.choices[0].message.content  # ✅ FIXED

    return call



# Prompt Template

INTENT_CITY_PROMPT = """
You are a text parsing assistant.
Your job is to output only a JSON object with two keys: "intent" and "cities".

Rules:
1. "intent" = "get_weather" if the user asks anything about weather.
2. "cities" = list of city names detected (capitalize properly).
3. Output ONLY valid JSON.

User message:
{user_input}
"""

# intent_template = PromptTemplate(
#     input_variables=["user_input"],
#     template=INTENT_CITY_PROMPT
# )


# # Intent & City Extraction (with fallback)

# def extract_intent_and_cities(user_text: str, llm=None) -> IntentCityModel:
#     lower = user_text.lower()

#     WEATHER_KEYWORDS = [
#         "weather", "temperature", "temp", "forecast", "rain",
#         "humidity", "humid", "heat", "cold", "climate",
#         "snow", "wind", "sunny", "cloud"
#     ]

#     # No weather words → no LLM call
#     if not any(w in lower for w in WEATHER_KEYWORDS):
#         return IntentCityModel(intent="other", cities=[])

#     # Weather-related → call LLM
#     llm = llm or get_llm()
#     chain = LLMChain(llm=llm, prompt=intent_template)

#     try:
#         raw = chain.run({"user_input": user_text})
#         parsed = json.loads(raw.strip())

#         # Use Pydantic v2 validation
#         return IntentCityModel.model_validate(parsed)

#     except Exception:
#         # Fallback regex extraction
#         fallback_cities = fallback_extract_cities(user_text)
#         return IntentCityModel(intent="get_weather", cities=fallback_cities)


# Using groq direct call for Intent & City Extraction

def extract_intent_and_cities(user_text: str, llm=None) -> IntentCityModel:
    lower = user_text.lower()

    WEATHER_KEYWORDS = [
        "weather", "temperature", "temp", "forecast", "rain",
        "humidity", "humid", "heat", "cold", "climate",
        "snow", "wind", "sunny", "cloud"
    ]

    # No weather words → no LLM call
    if not any(w in lower for w in WEATHER_KEYWORDS):
        return IntentCityModel(intent="other", cities=[])

    llm = llm or get_llm()

    # Direct Groq prompt (replaces LLMChain)
    prompt = f"""
You are a text parsing assistant.
Your job is to output only a JSON object with two keys: "intent" and "cities".

Rules:
1. "intent" = "get_weather" if the user asks anything about weather.
2. "cities" = list of detected city names in proper capitalization.
3. Output ONLY valid JSON (no extra text).

User message:
{user_text}
"""

    try:
        raw = llm(prompt)
        parsed = json.loads(raw.strip())
        return IntentCityModel.model_validate(parsed)

    except Exception:
        # Fallback regex extraction
        fallback_cities = fallback_extract_cities(user_text)
        return IntentCityModel(intent="get_weather", cities=fallback_cities)



# Fallback city extractor

def fallback_extract_cities(text: str) -> List[str]:
    text = text.strip().lower()

    # Capture text after "in", "at", "for"
    pattern = r'\b(?:in|at|of|for)\s+([a-z][a-z\s,]+)'
    matches = re.findall(pattern, text)

    parts = re.split(r'and|,', text)
    candidates = []

    for p in parts:
        words = p.strip().split()
        if words:
            candidates.append(words[-1])

    all_cities = set()
    for item in matches + candidates:
        item = item.strip().title()
        if len(item) > 2:
            all_cities.add(item)

    return list(all_cities)


# Weather API Caller (returns WeatherResult model)

def fetch_weather_for_city(city: str) -> WeatherResult:
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        res = requests.get(url, timeout=10)
        data = res.json()

        if res.status_code != 200:
            msg = data.get("message", "unknown error")
            return WeatherResult(
                city=city,
                weather=f"data not available ({msg})",
                error=True,
                message=msg
            )

        if "main" not in data or "weather" not in data:
            return WeatherResult(
                city=city,
                weather="data not available (missing info)",
                error=True,
                message="missing keys"
            )

        desc = data["weather"][0]["description"].capitalize()
        temp = data["main"].get("temp")
        feels = data["main"].get("feels_like")
        hum = data["main"].get("humidity")

        if None in (temp, feels, hum):
            return WeatherResult(
                city=city,
                weather="data not available (missing main info)",
                error=True
            )

        weather_str = f"{desc}, {temp}°C (feels like {feels}°C), humidity {hum}%"

        return WeatherResult(
            city=city,
            weather=weather_str
        )

    except Exception as e:
        return WeatherResult(
            city=city,
            weather="data not available (error)",
            error=True,
            message=str(e)
        )


# Final Reply Generator

def generate_final_reply(llm, weather_data: List[WeatherResult], behaviours=None) -> str:
    behaviours = behaviours or []
    behaviour_text = ", ".join(behaviours) if behaviours else "neutral"

    city_block = "\n".join([f"{w.city}: {w.weather}" for w in weather_data])
    city_names = ", ".join([w.city for w in weather_data])

    prompt = f"""
    You are a {behaviour_text} weather chatbot.
    Below is authoritative weather data for specific cities.
    You must generate one short, natural paragraph summarizing the weather only for these cities: {city_names}.
    Do not add, guess, or omit any city.
    If weather for a city says 'data not available', mention that literally.

    Weather data:
    {city_block}

    Tone rules:
    - Keep the tone consistent with {behaviour_text} personality.
    - Avoid repetition, disclaimers, or extra suggestions.
    - Include temperature, conditions, and humidity where available.
    - Reply in 3–5 concise lines max.

    Now generate the final reply faithfully following these instructions.
    """

    response = llm(prompt)
    return response.strip()


# Non-weather reply

def generate_non_weather_reply(llm, user_input: str, behaviours=None) -> str:
    behaviours = behaviours or []
    behaviour_text = ", ".join(behaviours) if behaviours else "neutral"

    prompt = f"""
You are a {behaviour_text} assistant that ONLY provides weather information.

User message: "{user_input}"

Reply with ONE short sentence saying you only answer weather-related questions.
"""

    return llm(prompt).strip()


# Main controller

def handle_user_message(user_text: str, behaviours=None) -> str:
    behaviours = behaviours or []
    llm = get_llm()

    parsed = extract_intent_and_cities(user_text, llm=llm)

    if parsed.intent != "get_weather":
        return generate_non_weather_reply(llm, user_text, behaviours)

    if not parsed.cities:
        return "Sure! Please tell me which city you'd like to know the weather for."

    weather_results = [fetch_weather_for_city(c) for c in parsed.cities]

    return generate_final_reply(llm, weather_results, behaviours)
