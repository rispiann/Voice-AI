from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import importlib

load_dotenv()

fr = importlib.import_module("fr")

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Friday API is running!"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_msg = data.get('message', '')

        if not user_msg:
            return jsonify({'reply': 'Pesan kosong, coba ketik sesuatu!'}), 400

        reply = fr.friday_response(user_msg)

        return jsonify({'reply': reply})
    except Exception as e:
        print("Error:", e)
        return jsonify({'reply': f'❌ Terjadi kesalahan: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.getenv('FRIDAY_API_PORT', 5000))
    print(f"🚀 Friday API berjalan di http://localhost:{port}")
    app.run(host='0.0.0.0', port=port)
