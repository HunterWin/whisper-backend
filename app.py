# app.py
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os
import json

app = Flask(__name__)

print("🔁 Cargando modelo Whisper...")
model_size = "medium"  # Puedes ajustar según el plan de Railway, pero conviene algunos como basic, tiny y demás
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("✅ Modelo cargado.")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No se proporcionó archivo de audio"}), 400

    audio_file = request.files["audio"]
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", audio_file.filename)
    audio_file.save(file_path)
    print(f"📥 Audio guardado: {file_path}")

    try:
        segments, info = model.transcribe(file_path, language="es")
        full_text = "".join([segment.text for segment in segments])
        print("✅ Transcripción completa.")
    except Exception as e:
        print(f"❌ Error al transcribir: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(file_path)
        print("🧹 Archivo eliminado.")

    return app.response_class(
        response=json.dumps({"text": full_text}, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

