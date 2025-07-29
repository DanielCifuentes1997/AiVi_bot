# -----------------------------------------------------------------------------
# AiVi DIAN - Servidor Backend v5.0 (Fase 5 - Análisis de Sentimiento)
#
# - Se implementa el análisis de sentimiento en la ruta /chat.
# - El prompt de Gemini ahora se adapta al estado de ánimo del usuario.
# -----------------------------------------------------------------------------

import os
import sqlite3
import shutil
import fitz
from datetime import datetime
import face_recognition
import numpy as np
import base64
import io
import markdown
from PIL import Image
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from flask_session import Session
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. CONFIGURACIÓN GENERAL ---
load_dotenv()
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "una-clave-secreta-por-defecto")
Session(app)

DB_NAME = "aivi_dian.db"
RUT_TEMPLATE_PATH = "RUT_editable.pdf"
USER_DOCS_PATH = "documentos_rut"
PDF_FIELD_MAP = { "nombre": "Primer nombre", "apellido": "Primer apellido", "cedula": "Número de Identificación", "nit": "5. Número de Identificación Tributaria", "direccion": "41. Dirección principal", "email": "42. Correo electrónico", "firma": "Firma del solicitante" }

# --- 2. CEREBRO CONVERSACIONAL (GEMINI) ---
def extract_pdf_text(pdf_path):
    if not os.path.exists(pdf_path): return f"Error: No se encontró el archivo PDF: {pdf_path}"
    try:
        doc = fitz.open(pdf_path); text = "".join(page.get_text() for page in doc); doc.close(); return text
    except Exception as e: return f"Error al leer PDF: {e}"

KNOWLEDGE_PDF = "preguntas_frecuentes_varias.pdf"
KNOWLEDGE_BASE = extract_pdf_text(KNOWLEDGE_PDF)

try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: raise ValueError("Variable GEMINI_API_KEY no encontrada en .env")
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    print(f"Error configurando Gemini: {e}")
    model_gemini = None

# --- 3. LÓGICA DE LA BASE DE DATOS Y USUARIOS ---
# ... (Todo el código de setup_database, load_known_users, login_vision, register_user, 
# get_user_status, logout, create_rut, y update_rut se mantiene sin cambios) ...
known_face_encodings = []; known_face_ids = []
def setup_database(): #...
    conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS usuarios (id INTEGER PRIMARY KEY, nombre_completo TEXT, cedula TEXT UNIQUE, email TEXT UNIQUE, huella_facial BLOB, ruta_rut_pdf TEXT, fecha_modificacion TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS calificaciones (id INTEGER PRIMARY KEY, user_cedula TEXT, rating INTEGER, timestamp TEXT)")
    conn.commit(); conn.close()
def load_known_users(): #...
    global known_face_encodings, known_face_ids; known_face_encodings, known_face_ids = [], []
    print("Cargando usuarios..."); conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
    try:
        cursor.execute("SELECT cedula, huella_facial FROM usuarios")
        for cedula, encoding_blob in cursor.fetchall():
            encoding = np.frombuffer(encoding_blob, dtype=np.float64); known_face_encodings.append(encoding); known_face_ids.append(cedula)
        print(f"-> {len(known_face_ids)} usuarios cargados.")
    except sqlite3.OperationalError: print("Tabla 'usuarios' no encontrada.")
    conn.close()
@app.route("/login_vision", methods=['POST'])
def handle_login_vision(): #...
    try:
        data = request.json
        if 'image' not in data: return jsonify({'error': 'No se proporcionó imagen'}), 400
        image_data = base64.b64decode(data['image'].split(',')[1]); frame = np.array(Image.open(io.BytesIO(image_data)))
        face_locations = face_recognition.face_locations(frame); face_encodings = face_recognition.face_encodings(frame, face_locations)
        if not face_encodings: return jsonify({'status': 'no_face_detected'})
        face_encoding = face_encodings[0]; matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        user_id_cedula = "Desconocido"; user_name = "Desconocido"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                user_id_cedula = known_face_ids[best_match_index]
                conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
                cursor.execute("SELECT nombre_completo FROM usuarios WHERE cedula = ?", (user_id_cedula,))
                result = cursor.fetchone()
                conn.close()
                if result:
                    user_name = result[0]; session['user_id'] = user_id_cedula; session['user_name'] = user_name; session['chat_history'] = []
                    print(f"SESIÓN INICIADA para: {user_name} ({user_id_cedula})")
        return jsonify({'status': 'face_analyzed', 'user_id': user_id_cedula, 'user_name': user_name})
    except Exception as e: return jsonify({'error': str(e)}), 500
@app.route('/register_user', methods=['POST'])
def handle_register_user(): #...
    try:
        data = request.json; name = data.get('name'); cedula = data.get('cedula'); email = data.get('email'); images_data = data.get('images')
        if not all([name, cedula, email, images_data]): return jsonify({'error': 'Faltan datos'}), 400
        face_encodings = []
        for img_data in images_data:
            img_bytes = base64.b64decode(img_data.split(',')[1]); img = np.array(Image.open(io.BytesIO(img_bytes))); current_encoding = face_recognition.face_encodings(img)
            if current_encoding: face_encodings.append(current_encoding[0])
        if not face_encodings: return jsonify({'error': 'No se encontró cara clara'}), 400
        average_encoding = np.mean(face_encodings, axis=0); encoding_bytes = average_encoding.tobytes()
        conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
        cursor.execute("INSERT INTO usuarios (nombre_completo, cedula, email, huella_facial) VALUES (?, ?, ?, ?)", (name, cedula, email, encoding_bytes))
        conn.commit(); conn.close()
        load_known_users(); print(f"NUEVO USUARIO REGISTRADO: {name} ({cedula})!"); return jsonify({'status': 'success', 'message': f'Usuario {name} registrado.'})
    except sqlite3.IntegrityError: return jsonify({'error': 'La cédula o el email ya están registrados.'}), 400
    except Exception as e: print(f"Error en registro: {e}"); return jsonify({'error': str(e)}), 500
@app.route("/get_user_status")
def get_user_status(): #...
    if 'user_id' not in session: return jsonify({'error': 'No hay sesión activa'}), 401
    user_id_cedula = session['user_id']; conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
    cursor.execute("SELECT ruta_rut_pdf FROM usuarios WHERE cedula = ?", (user_id_cedula,)); result = cursor.fetchone(); conn.close()
    if result and result[0]: return jsonify({'has_rut': True})
    else: return jsonify({'has_rut': False})
@app.route("/logout")
def logout(): #...
    session.clear(); return jsonify({'status': 'success', 'message': 'Sesión cerrada.'})
@app.route('/create_rut', methods=['POST'])
def handle_create_rut(): #...
    if 'user_id' not in session: return jsonify({'error': 'No hay sesión activa'}), 401
    try:
        data = request.json; user_cedula = session['user_id']; user_name = session['user_name']
        if not os.path.exists(USER_DOCS_PATH): os.makedirs(USER_DOCS_PATH)
        new_rut_filename = f"RUT_{user_name.replace(' ', '_')}_{user_cedula}.pdf"; new_rut_path = os.path.join(USER_DOCS_PATH, new_rut_filename)
        shutil.copy(RUT_TEMPLATE_PATH, new_rut_path)
        doc = fitz.open(new_rut_path)
        for key, value in data.items():
            pdf_field_name = PDF_FIELD_MAP.get(key)
            if pdf_field_name:
                for page in doc:
                    for field in page.widgets():
                        if field.field_name == pdf_field_name: field.field_value = str(value); field.update()
        firma_field_name = PDF_FIELD_MAP.get("firma")
        if firma_field_name:
            for page in doc:
                for field in page.widgets():
                    if field.field_name == firma_field_name: field.field_value = f"{user_name} (Firma Electrónica)"; field.update()
        doc.saveIncr(); doc.close()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
        cursor.execute("UPDATE usuarios SET ruta_rut_pdf = ?, fecha_modificacion = ? WHERE cedula = ?", (new_rut_path, current_time, user_cedula))
        conn.commit(); conn.close()
        print(f"RUT CREADO para {user_name}"); return jsonify({'status': 'success', 'message': 'RUT creado correctamente.'})
    except Exception as e: print(f"Error creando RUT: {e}"); return jsonify({'error': str(e)}), 500
@app.route('/update_rut', methods=['POST'])
def handle_update_rut(): #...
    if 'user_id' not in session: return jsonify({'error': 'No hay sesión activa'}), 401
    try:
        data = request.json; field_to_update = data.get('field'); new_value = data.get('value'); user_cedula = session['user_id']; user_name = session['user_name']
        if not field_to_update or new_value is None: return jsonify({'error': 'Faltan datos'}), 400
        conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
        cursor.execute("SELECT ruta_rut_pdf FROM usuarios WHERE cedula = ?", (user_cedula,)); result = cursor.fetchone()
        if not result or not result[0]: conn.close(); return jsonify({'error': 'Usuario sin RUT para actualizar'}), 404
        rut_path = result[0]; doc = fitz.open(rut_path); pdf_field_name = PDF_FIELD_MAP.get(field_to_update)
        if not pdf_field_name: doc.close(); conn.close(); return jsonify({'error': f'Campo "{field_to_update}" inválido'}), 400
        field_found = False
        for page in doc:
            for field in page.widgets():
                if field.field_name == pdf_field_name: field.field_value = new_value; field.update(); field_found = True; break
            if field_found: break
        if not field_found: doc.close(); conn.close(); return jsonify({'error': f'Campo "{pdf_field_name}" no encontrado en PDF'}), 500
        firma_field_name = PDF_FIELD_MAP.get("firma")
        if firma_field_name:
            for page in doc:
                for field in page.widgets():
                    if field.field_name == firma_field_name: field.field_value = f"{user_name} (Firma Electrónica)"; field.update()
        doc.saveIncr(); doc.close()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("UPDATE usuarios SET fecha_modificacion = ? WHERE cedula = ?", (current_time, user_cedula)); conn.commit(); conn.close()
        print(f"RUT de {user_name} actualizado. Campo: {field_to_update}"); return jsonify({'status': 'success', 'message': 'RUT actualizado.'})
    except Exception as e: print(f"Error actualizando RUT: {e}"); return jsonify({'error': str(e)}), 500

# --- 4. RUTAS DE CHAT Y CALIFICACIÓN (MODIFICADAS) ---
@app.route('/chat', methods=['POST'])
def handle_chat_final():
    if 'user_id' not in session: return jsonify({'error': 'No hay sesión activa'}), 401
    if not model_gemini: return jsonify({'error': 'Modelo IA no configurado'}), 500
    
    prompt = request.json.get("prompt")
    if not prompt: return jsonify({'error': 'No se recibió pregunta'}), 400

    if 'chat_history' not in session: session['chat_history'] = []
    
    # --- ANÁLISIS DE SENTIMIENTO ---
    sentiment = "neutro"
    try:
        sentiment_prompt = f'Analiza el sentimiento del siguiente texto. Responde únicamente con una de estas tres palabras: positivo, negativo, neutro. Texto: "{prompt}"'
        sentiment_response = model_gemini.generate_content(sentiment_prompt)
        detected_sentiment = sentiment_response.text.strip().lower()
        if detected_sentiment in ['positivo', 'negativo', 'neutro']:
            sentiment = detected_sentiment
    except Exception as e:
        print(f"Error al analizar sentimiento: {e}")

    # --- PROMPT ENRIQUECIDO ---
    history_context = "\n".join([f"Usuario: {h['prompt']}\nAsistente: {h['response']}" for h in session['chat_history'][-3:]])
    
    specialized_prompt = f"""
    Eres AiVi, un asistente experto de la DIAN. Eres amable, cortés y muy empático.
    
    **Análisis de la Conversación:**
    - Sentimiento del último mensaje del usuario: **{sentiment}**
    - Historial de la conversación: {history_context}

    **TUS INSTRUCCIONES DE COMPORTAMIENTO:**
    1.  **ADAPTA TU TONO:** Ajusta tu estilo de respuesta al sentimiento del usuario.
        - Si el sentimiento es **negativo**, tu tono debe ser especialmente empático y comprensivo. Ofrece ayuda de forma proactiva.
        - Si el sentimiento es **positivo**, responde de manera amigable y entusiasta.
        - Si el sentimiento es **neutro**, mantén un tono profesional, claro y directo.
    2.  **USA TU BASE DE CONOCIMIENTO:** Tu única fuente de verdad es el siguiente documento. Responde basándote exclusivamente en él.
    3.  **SÉ PROACTIVO:** Si la respuesta no está en el documento, no digas "no sé". En su lugar, responde cortésmente: "No tengo información sobre ese tema en mi base de conocimiento, pero puedo ayudarte con otros trámites del RUT."

    --- DOCUMENTO DE CONOCIMIENTO ---
    {KNOWLEDGE_BASE}
    --- FIN DEL DOCUMENTO ---

    Ahora, responde a la siguiente pregunta del usuario aplicando estrictamente tus instrucciones:
    Usuario: {prompt}
    """
    try:
        response = model_gemini.generate_content(specialized_prompt)
        response_text = response.text
        
        session['chat_history'].append({"prompt": prompt, "response": response_text, "sentiment": sentiment})
        session.modified = True
        return jsonify({'response': markdown.markdown(response_text)})
    except Exception as e:
        print(f"Error en Gemini: {e}")
        return jsonify({'error': f'Error al contactar al modelo de IA: {e}'}), 500

@app.route('/rate_chat', methods=['POST'])
def rate_chat():
    # ... (código sin cambios) ...
    if 'user_id' not in session: return jsonify({'error': 'No hay sesión activa'}), 401
    rating = request.json.get('rating')
    if not rating: return jsonify({'error': 'No se recibió calificación'}), 400
    user_cedula = session['user_id']; timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
    cursor.execute("INSERT INTO calificaciones (user_cedula, rating, timestamp) VALUES (?, ?, ?)", (user_cedula, rating, timestamp))
    conn.commit(); conn.close()
    print(f"Calificación recibida: {rating} para el usuario {user_cedula}")
    return jsonify({'status': 'success'})

# --- 5. RUTA PRINCIPAL Y ARRANQUE ---
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    setup_database()
    load_known_users()
    print(f"Servidor AiVi DIAN listo.")
    app.run(host='0.0.0.0', port=5000, debug=True)
