"""Send one test question to Gemini. Run from the project root:

    python tests\\try_gemini.py
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY", "").strip()
model_name = os.getenv("GEMINI_MODEL", "").strip()

if not api_key or api_key == "paste-your-key-here":
    sys.exit("GEMINI_API_KEY is missing. Put your key in the .env file.")
if not model_name:
    sys.exit("GEMINI_MODEL is missing. Add it to the .env file.")

print(f"Key found: yes (length {len(api_key)}) | model: {model_name}")

client = genai.Client(api_key=api_key)
question = "In one sentence, what is a university admission policy?"

start = time.time()
try:
    response = client.models.generate_content(model=model_name, contents=question)
except Exception as exc:
    message = str(exc).replace(api_key, "[hidden]")  # never print the key
    print(f"Request failed ({type(exc).__name__}): {message[:500]}")
    sys.exit(1)

print(f"Answered in {time.time() - start:.1f} seconds")
print("Answer:", response.text or "(the model returned no text)")