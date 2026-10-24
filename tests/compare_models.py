"""Compare Gemini models for speed and RAG-style answers.

Run from the project root:   python tests\\compare_models.py
Uses only fictional sample text, so nothing private is sent.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
api_key = os.getenv("GEMINI_API_KEY", "").strip()
if not api_key or api_key in ("paste-your-key-here", "your-api-key-here"):
    sys.exit("GEMINI_API_KEY is missing or still a placeholder in .env")

MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]

SYSTEM = (
    "You answer student questions using ONLY the numbered passages provided. "
    "Cite the source like (file, page N). If the passages do not contain the "
    "answer, say that the documents do not provide enough information."
)
PASSAGES = (
    "[1] sample_exam_rules.pdf, page 1:\n"
    "Students must have at least 75 percent attendance in a course to be allowed "
    "to sit the final examination. Students below this limit receive an F grade "
    "in the course.\n\n"
    "[2] sample_exam_rules.pdf, page 2:\n"
    "The minimum passing grade is D, which requires 50 percent marks. A student "
    "may re-take a failed course at most twice."
)
QUESTIONS = [
    "How much attendance do I need to sit the final exam?",
    "What is the hostel fee?",
]

print("ThinkingConfig fields:", sorted(types.ThinkingConfig.model_fields))
print("HttpOptions fields:", sorted(types.HttpOptions.model_fields))

client = genai.Client(api_key=api_key)


def run(model, question, thinking=None):
    start = time.time()
    try:
        config_args = {"system_instruction": SYSTEM, "temperature": 0.2}
        if thinking:
            config_args["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
        response = client.models.generate_content(
            model=model,
            contents=f"Passages:\n{PASSAGES}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(**config_args),
        )
    except Exception as exc:
        message = str(exc).replace(api_key, "[hidden]")
        print(f"  FAILED after {time.time() - start:.1f}s: {type(exc).__name__} "
              f"code={getattr(exc, 'code', None)} | {message[:200]}")
        return
    usage = getattr(response, "usage_metadata", None)
    thoughts = getattr(usage, "thoughts_token_count", None)
    answer = " ".join((response.text or "(no text returned)").split())
    print(f"  {time.time() - start:.1f}s | thought tokens: {thoughts} | {answer[:300]}")


for model in MODELS:
    print(f"\n=== {model} (default thinking) ===")
    for question in QUESTIONS:
        print(f" Q: {question}")
        run(model, question)

print("\n=== Lower thinking settings ===")
print(" gemini-3.8-flash, thinking_level=low")
run("gemini-3.8-flash", QUESTIONS[0], thinking="low")
print(" gemini-3.6-flash, thinking_level=minimal")
run("gemini-3.6-flash", QUESTIONS[0], thinking="minimal")