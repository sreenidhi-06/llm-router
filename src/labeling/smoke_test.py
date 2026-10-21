import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
client = Groq(api_key=os.environ["GROQ_API_KEY"])

for model in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=512, reasoning_effort="low",
        messages=[{"role": "user", "content":
                   "Reply with only the letter of the correct answer.\n"
                   "What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6"}])
    print(model, "->", repr(r.choices[0].message.content), "| tokens:", r.usage.total_tokens)