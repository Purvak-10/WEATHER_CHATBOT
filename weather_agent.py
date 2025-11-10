import os
import json
import requests
import re
from dotenv import load_dotenv
from typing import List, Optional, Literal
from pydantic import BaseModel, field_validator
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq   


load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not OPENWEATHER_API_KEY:
    raise RuntimeError("OPENWEATHER_API_KEY not found in .env")


#Pydantic 
class IntentCityModel(BaseModel):
    intent: Literal["get_weather", "other"]
    cities: List[str] = []

    @field_validator("cities", mode="before")
    def normalize_city_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            cleaned = [x.strip().title() for x in value if isinstance(x, str) and x.strip()]
            return cleaned
        if isinstance(value, str):
            parts = re.split(r",|\band\b", value)
            cleaned = [p.strip().title() for p in parts if p.strip()]
            return cleaned
        return []

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
        return v.strip().title() if isinstance(v, str) else str(v)

    @field_validator("weather", mode="before")
    def normalize_weather(cls, v):
        return "data not available" if v is None else str(v).strip()


#LangChain LLM Helper
def get_llm():
    """Return a LangChain ChatGroq LLM."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )


#Prompt
INTENT_CITY_PROMPT = """
You are an expert text parser.
Extract the user's intent and cities.

Rules:
- "intent" = "get_weather" if the user asks about weather.
- "cities" = exactly the detected city names (properly capitalized)
- Output ONLY valid JSON. No extra text.

User message: {user_input}
"""


intent_template = PromptTemplate(
    input_variables=["user_input"],
    template=INTENT_CITY_PROMPT
)


#Intent & City Extraction
def extract_intent_and_cities(user_text: str, llm=None) -> IntentCityModel:
    lower = user_text.lower()

    WEATHER_WORDS = [
        "weather", "temperature", "temp", "forecast", "rain",
        "humidity", "humid", "heat", "cold", "climate",
        "snow", "wind", "sunny", "cloud"
    ]

    #Not weather then no LLM call
    if not any(w in lower for w in WEATHER_WORDS):
        return IntentCityModel(intent="other", cities=[])

    llm = llm or get_llm()
    chain = LLMChain(llm=llm, prompt=intent_template)

    try:
        raw_json = chain.run({"user_input": user_text})
        parsed = json.loads(raw_json)
        return IntentCityModel.model_validate(parsed)

    except Exception:
        return IntentCityModel(
            intent="get_weather",
            cities=fallback_extract_cities(user_text)
        )


#Fallback Regex City Extractor
def fallback_extract_cities(text: str) -> List[str]:
    text = text.strip().lower()
    matches = re.findall(r'\b(?:in|at|of|for)\s+([a-z][a-z\s,]+)', text)

    parts = re.split(r'and|,', text)
    candidates = []

    for part in parts:
        words = part.strip().split()
        if words:
            candidates.append(words[-1])

    all_cities = set()
    for item in matches + candidates:
        item = item.strip().title()
        if len(item) > 2:
            all_cities.add(item)

    return list(all_cities)


#WEATHER API
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
            return WeatherResult(city=city, weather=f"data not available ({msg})", error=True)

        desc = data["weather"][0]["description"].capitalize()
        temp = data["main"].get("temp")
        feels = data["main"].get("feels_like")
        hum = data["main"].get("humidity")

        if None in (temp, feels, hum):
            return WeatherResult(city=city, weather="data not available (missing info)", error=True)

        weather_str = f"{desc}, {temp}°C (feels like {feels}°C), humidity {hum}%"
        return WeatherResult(city=city, weather=weather_str)

    except Exception as e:
        return WeatherResult(city=city, weather="data not available", error=True, message=str(e))


#FINAL REPLY GENERATOR
def generate_final_reply(llm, weather_data: List[WeatherResult], behaviours=None) -> str:
    behaviours = behaviours or []
    behaviour_text = ", ".join(behaviours) if behaviours else "neutral"

    city_block = "\n".join([f"{w.city}: {w.weather}" for w in weather_data])
    city_names = ", ".join([w.city for w in weather_data])

    prompt = f"""
You are a {behaviour_text} weather assistant.

Summarize the weather for these cities ONLY:
{city_names}

Weather data:
{city_block}

Rules:
- No guessing
- Mention "data not available" exactly when needed
- One short natural paragraph
"""

    response = llm.invoke(prompt)
    return response.content.strip()


#NON-WEATHER REPLY
def generate_non_weather_reply(llm, user_input: str, behaviours=None):
    behaviours = behaviours or []
    behaviour_text = ", ".join(behaviours) if behaviours else "neutral"

    prompt = f"""
You are a {behaviour_text} assistant.

User message: "{user_input}"

Reply with ONE sentence saying:
"I only provide weather information."
"""

    response = llm.invoke(prompt)
    return response.content.strip()


#MAIN CONTROLLER
def handle_user_message(user_text: str, behaviours=None) -> str:
    behaviours = behaviours or []
    llm = get_llm()

    parsed = extract_intent_and_cities(user_text, llm=llm)

    if parsed.intent != "get_weather":
        return generate_non_weather_reply(llm, user_text, behaviours)

    if not parsed.cities:
        return "Sure! Please tell me which city you'd like weather for."

    weather_results = [fetch_weather_for_city(c) for c in parsed.cities]

    return generate_final_reply(llm, weather_results, behaviours)

