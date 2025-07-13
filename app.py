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
    return "🧠 Groq Chatbot & Vision API is running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages")
    user_msg = data.get("message", "")

    if messages:
        api_payload = {
            "model": "deepseek-r1-distill-llama-70b",
            "messages": messages
        }
    elif user_msg:
        api_payload = {
            "model": "deepseek-r1-distill-llama-70b",
            "messages": [{"role": "user", "content": user_msg}]
        }
    else:
        return jsonify({"error": "No message provided"}), 400

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        res = requests.post(GROQ_URL, headers=headers, json=api_payload, timeout=20)
        res.raise_for_status()
        reply = res.json()["choices"][0]["message"]["content"]
        return jsonify({"reply": reply.strip()})
    except requests.exceptions.Timeout:
        return jsonify({"error": "Groq API timeout"}), 504
    except Exception as e:
        return jsonify({"error": "Groq API failed", "details": str(e)}), 500

@app.route("/vision", methods=["POST"])
def vision():
    data = request.json
    image_url = data.get("image_url")
    prompt = data.get("prompt", "इस फोटो में क्या दिख रहा है?")

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    api_payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "temperature": 0.5,
        "max_tokens": 1024
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        res = requests.post(GROQ_URL, headers=headers, json=api_payload, timeout=30)
        res.raise_for_status()
        reply = res.json()["choices"][0]["message"]["content"]
        return jsonify({"reply": reply.strip()})
    except requests.exceptions.Timeout:
        return jsonify({"error": "Groq Vision API timeout"}), 504
    except Exception as e:
        return jsonify({"error": "Groq Vision API failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
