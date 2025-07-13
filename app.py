from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
CORS(app)

API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.route("/", methods=["GET"])
def home():
    return "🧠 Groq Unified Chat + Vision API is running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json

    # ✅ Input options
    messages = data.get("messages")
    message = data.get("message", "")
    image_url = data.get("image_url")
    image_base64 = data.get("image_base64")  # Base64 string with proper prefix (data:image/jpeg;base64,...)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # ✅ If Vision: image_url or image_base64
    if (image_url or image_base64) and message:
        image_data = {"url": image_url} if image_url else {"url": image_base64}

        api_payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                        {"type": "image_url", "image_url": image_data}
                    ]
                }
            ],
            "temperature": 0.5,
            "max_tokens": 1024
        }

    # ✅ Multi-turn text only
    elif messages:
        api_payload = {
            "model": "deepseek-r1-distill-llama-70b",
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1024
        }

    # ✅ Single-turn text
    elif message:
        api_payload = {
            "model": "deepseek-r1-distill-llama-70b",
            "messages": [{"role": "user", "content": message}],
            "temperature": 0.5,
            "max_tokens": 1024
        }

    else:
        return jsonify({"error": "No valid input provided"}), 400

    # 🔁 API Call
    try:
        res = requests.post(GROQ_URL, headers=headers, json=api_payload, timeout=30)
        res.raise_for_status()
        reply = res.json()["choices"][0]["message"]["content"]
        return jsonify({"reply": reply.strip()})
    except requests.exceptions.Timeout:
        return jsonify({"error": "Groq API timeout"}), 504
    except Exception as e:
        return jsonify({"error": "Groq API failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)