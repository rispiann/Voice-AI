from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import importlib

# 🧩 Load environment variables (.env)
load_dotenv()

# 🧠 Import Friday AI (pastikan fr.py ada di folder yang sama)
fr = importlib.import_module("fr")

app = Flask(__name__)
CORS(app)

# 🚀 Route tes untuk memastikan server hidup
@app.route('/')
def home():
    return jsonify({"message": "Friday API is running!"})

# 💬 Route utama: menerima chat dari frontend dan balas pakai Friday AI
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_msg = data.get('message', '')

        if not user_msg:
            return jsonify({'reply': 'Pesan kosong, coba ketik sesuatu!'}), 400

        # 🔮 Panggil fungsi dari fr.py (ganti sesuai struktur Friday kamu)
        reply = fr.friday_response(user_msg)

        return jsonify({'reply': reply})
    except Exception as e:
        print("Error:", e)
        return jsonify({'reply': f'❌ Terjadi kesalahan: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.getenv('FRIDAY_API_PORT', 5000))
    print(f"🚀 Friday API berjalan di http://localhost:{port}")
    app.run(host='0.0.0.0', port=port)
