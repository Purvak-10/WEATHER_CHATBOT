# import streamlit as st
# from dotenv import load_dotenv
# import time

# from weather_agent import (
#     extract_intent_and_cities,
#     fetch_weather_for_city,
#     generate_final_reply,
#     generate_non_weather_reply,
#     get_llm
# )

# load_dotenv()

# st.set_page_config(page_title="Weather Chatbot", layout="centered")
# st.title("Weather Chatbot (LangChain + Llama local + OpenWeather)")

# st.markdown(
#     """
# Enter a message (e.g. "What's the weather in Delhi and Kathmandu?")  
# Choose one or more behaviours to change how the bot responds.
# """
# )

# # Available behaviours
# ALL_BEHAVIOURS = [
#     "friendly", "concise", "detailed", "humorous", "technical",
#     "poetic", "emoji", "cheerful", "empathetic"
# ]

# with st.sidebar:
#     st.header("Behaviour")
#     st.write("Choose chatbot behaviours:")
#     chosen_behaviours = st.multiselect(
#         "Behaviours", 
#         ALL_BEHAVIOURS, 
#         default=["friendly", "concise"]
#     )
#     st.markdown("---")

#     if st.button("Show example prompts"):
#         st.markdown(
#             """
# **Intent / City extractor prompt example:**  
# `{ "intent": "get_weather", "cities": ["Delhi", "Kathmandu"] }`

# **Response prompt:**  
# Uses chosen behaviours to generate a natural weather summary.
# """
#         )

# # Session state for chat
# if "chat" not in st.session_state:
#     st.session_state.chat = []

# def add_message(role, text):
#     st.session_state.chat.append({"role": role, "text": text})

# # Display chat history
# for msg in st.session_state.chat:
#     if msg["role"] == "user":
#         st.markdown(f"**You:** {msg['text']}")
#     else:
#         st.markdown(f"**Bot:** {msg['text']}")

# # Input area
# user_input = st.text_input("Message", key="user_input")
# submit = st.button("Send")

# if submit and user_input:
#     add_message("user", user_input)

#     # Understanding intent
#     with st.spinner("Understanding your message..."):
#         llm = get_llm()
#         parsed = extract_intent_and_cities(user_input, llm=llm)

#     intent = parsed.get("intent")
#     cities = parsed.get("cities", [])

#     # WEATHER INTENT
#     if intent == "get_weather":

#         # No city found → ask user
#         if not cities:
#             bot_text = (
#                 "I think you want to know about the weather, "
#                 "but I couldn't find any city names. "
#                 "Please tell me which city or cities (e.g., 'Delhi', 'Mumbai and Pune')."
#             )
#             add_message("bot", bot_text)
#             st.rerun()

#         # Fetch weather for all cities
#         weather_results = []
#         with st.spinner("Fetching weather for the mentioned cities..."):
#             for c in cities:
#                 time.sleep(0.2)  # smooth UI
#                 try:
#                     res = fetch_weather_for_city(c)
#                 except Exception as e:
#                     res = {"city": c, "error": True, "message": str(e)}
#                 weather_results.append(res)

#         # Generate final weather reply
#         with st.spinner("Generating response..."):
#             final_reply = generate_final_reply(llm, weather_results, chosen_behaviours)

#         add_message("bot", final_reply)
#         st.rerun()

#     # NON-WEATHER INTENT
#     else:
#         with st.spinner("Generating reply..."):
#             response = generate_non_weather_reply(llm, user_input, chosen_behaviours)

#         add_message("bot", response)
#         st.rerun()




# using pydantic
# app.py
import streamlit as st
from dotenv import load_dotenv
import time

from weather_agent import (
    extract_intent_and_cities,
    fetch_weather_for_city,
    generate_final_reply,
    generate_non_weather_reply,
    get_llm,
    WeatherResult,
)

load_dotenv()

st.set_page_config(page_title="Weather Chatbot", layout="centered")
st.title("Weather Chatbot (LangChain + Llama Local + OpenWeather)")

st.markdown(
    """
Enter a message (e.g., "What's the weather in Delhi and Kathmandu?")  
Choose one or more behaviours to change how the bot responds.
"""
)

# -----------------------------------------------
# ✅ Available behaviours
# -----------------------------------------------
ALL_BEHAVIOURS = [
    "friendly", "concise", "detailed", "humorous", "technical",
    "poetic", "emoji", "cheerful", "empathetic"
]

with st.sidebar:
    st.header("Behaviour Settings")
    chosen_behaviours = st.multiselect(
        "Select behaviours:",
        ALL_BEHAVIOURS,
        default=["friendly", "concise"]
    )
    st.markdown("---")

    if st.button("Show example prompts"):
        st.write(
            """
**Intent/City Extractor Example:**

{ "intent": "get_weather", "cities": ["Delhi", "Kathmandu"] }
**Response Generator:**

Uses the chosen behaviours to adjust the writing style.
"""
        )

# -----------------------------------------------
# ✅ Session state to store chat history
# -----------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

def add_message(role: str, text: str):
    st.session_state.chat.append({"role": role, "text": text})

# -----------------------------------------------
# ✅ Display existing chat
# -----------------------------------------------
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Bot:** {msg['text']}")

# -----------------------------------------------
# ✅ User input
# -----------------------------------------------
user_input = st.text_input("Message", key="user_input")
submit = st.button("Send")

if submit and user_input:
    add_message("user", user_input)

    llm = get_llm()

    # ---------------------------------------------------
    # ✅ Step 1: Intent + city extraction
    # ---------------------------------------------------
    with st.spinner("Understanding your message..."):
        parsed = extract_intent_and_cities(user_input, llm=llm)

    intent = parsed.intent
    cities = parsed.cities

    # ---------------------------------------------------
    # ✅ NON-WEATHER INTENT
    # ---------------------------------------------------
    if intent != "get_weather":
        with st.spinner("Generating reply..."):
            response = generate_non_weather_reply(llm, user_input, chosen_behaviours)

        add_message("bot", response)
        st.rerun()

    # ---------------------------------------------------
    # ✅ WEATHER INTENT BUT NO CITIES
    # ---------------------------------------------------
    if intent == "get_weather" and not cities:
        bot_text = (
            "I think you're asking about the weather, "
            "but I couldn't detect any city names. "
            "Please tell me which city or cities (e.g., 'Delhi', 'Mumbai and Pune')."
        )
        add_message("bot", bot_text)
        st.rerun()

    # ---------------------------------------------------
    # ✅ WEATHER INTENT + CITIES
    # ---------------------------------------------------
    weather_results: list[WeatherResult] = []

    with st.spinner("Fetching weather data..."):
        for c in cities:
            time.sleep(0.2)  # Smooth UI
            try:
                result = fetch_weather_for_city(c)
            except Exception as e:
                result = WeatherResult(
                    city=c,
                    weather="data not available (error)",
                    error=True,
                    message=str(e)
                )
            weather_results.append(result)

    # ---------------------------------------------------
    # ✅ Generate final natural reply
    # ---------------------------------------------------
    with st.spinner("Generating response..."):
        final_reply = generate_final_reply(llm, weather_results, chosen_behaviours)

    add_message("bot", final_reply)
    st.rerun()
