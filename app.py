# -----------------------------------------------------------------------------
# AiVi DIAN - Servidor Backend v5.2 (Versión Final con Guardado Robusto)
#
# - Se integra Flask-Mail para el envío de correos.
# - La ruta /rate_chat ahora envía una transcripción del chat al usuario.
# - Se corrige el método de guardado de PDF para máxima compatibilidad.
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
from flask_mail import Mail, Message
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

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
mail = Mail(app)

DB_NAME = "aivi_dian.db"
RUT_TEMPLATE_PATH = "RUT_editable.pdf"
USER_DOCS_PATH = "documentos_rut"
PDF_FIELD_MAP = { "nombre": "untitled14", "apellido": "untitled12", "cedula": "untitled1", "nit": "untitled1", "direccion": "untitled22", "email": "untitled23", "firma": "untitled95" }

# --- 2. CEREBRO CONVERSACIONAL (GEMINI) ---
def extract_pdf_text(pdf_path):
    if not os.path.exists(pdf_path): return f"Error: No se encontró el archivo PDF: {pdf_path}"
    try: doc = fitz.open(pdf_path); text = "".join(page.get_text() for page in doc); doc.close(); return text
    except Exception as e: return f"Error al leer PDF: {e}"
KNOWLEDGE_PDF = "preguntas_frecuentes_varias.pdf"
KNOWLEDGE_BASE = extract_pdf_text(KNOWLEDGE_PDF)
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: raise ValueError("Variable GEMINI_API_KEY no encontrada en .env")
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e: print(f"Error configurando Gemini: {e}"); model_gemini = None

# --- 3. LÓGICA DE LA BASE DE DATOS Y USUARIOS ---
known_face_encodings = []; known_face_ids = []
def setup_database():
    conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS usuarios (id INTEGER PRIMARY KEY, nombre_completo TEXT, cedula TEXT UNIQUE, email TEXT UNIQUE, huella_facial BLOB, ruta_rut_pdf TEXT, fecha_modificacion TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS calificaciones (id INTEGER PRIMARY KEY, user_cedula TEXT, rating INTEGER, timestamp TEXT)")
    conn.commit(); conn.close()
def load_known_users():
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
def handle_login_vision():
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
def handle_register_user():
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
def get_user_status():
    if 'user_id' not in session: return jsonify({'error': 'No hay sesión activa'}), 401
    user_id_cedula = session['user_id']; conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
    cursor.execute("SELECT ruta_rut_pdf FROM usuarios WHERE cedula = ?", (user_id_cedula,)); result = cursor.fetchone(); conn.close()
    if result and result[0]: return jsonify({'has_rut': True})
    else: return jsonify({'has_rut': False})
@app.route("/logout")
def logout():
    session.clear(); return jsonify({'status': 'success', 'message': 'Sesión cerrada.'})

# --- 4. GESTIÓN DEL TRÁMITE DEL RUT ---
@app.route('/create_rut', methods=['POST'])
def handle_create_rut():
    if 'user_id' not in session: return jsonify({'error': 'No hay sesión activa'}), 401
    try:
        data = request.json; user_cedula = session['user_id']; user_name = session['user_name']
        
        # --- CORRECCIÓN DE RUTA ABSOLUTA ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_template_path = os.path.join(script_dir, RUT_TEMPLATE_PATH)
        
        if not os.path.exists(USER_DOCS_PATH): os.makedirs(USER_DOCS_PATH)
        new_rut_filename = f"RUT_{user_name.replace(' ', '_')}_{user_cedula}.pdf"
        new_rut_path = os.path.join(USER_DOCS_PATH, new_rut_filename)
        
        shutil.copy(absolute_template_path, new_rut_path)
        
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
        
        # --- SOLUCIÓN DE GUARDADO ROBUSTO ---
        temp_path = new_rut_path + ".tmp"
        doc.save(temp_path, garbage=4, deflate=True)
        doc.close()
        os.remove(new_rut_path) # Borra el original copiado
        os.rename(temp_path, new_rut_path) # Renombra el nuevo al nombre final
        # --- FIN DE LA SOLUCIÓN ---
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
        cursor.execute("UPDATE usuarios SET ruta_rut_pdf = ?, fecha_modificacion = ? WHERE cedula = ?", (new_rut_path, current_time, user_cedula))
        conn.commit(); conn.close()
        
        print(f"RUT CREADO para {user_name}"); return jsonify({'status': 'success', 'message': 'RUT creado correctamente.'})

    except Exception as e: 
        print(f"Error creando RUT: {e}"); return jsonify({'error': str(e)}), 500
@app.route('/update_rut', methods=['POST'])
def handle_update_rut():
    if 'user_id' not in session: return jsonify({'error': 'No hay sesión activa'}), 401
    try:
        data = request.json; field_to_update = data.get('field'); new_value = data.get('value'); user_cedula = session['user_id']; user_name = session['user_name']
        if not field_to_update or new_value is None: return jsonify({'error': 'Faltan datos'}), 400
        
        conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
        cursor.execute("SELECT ruta_rut_pdf FROM usuarios WHERE cedula = ?", (user_cedula,)); result = cursor.fetchone()
        if not result or not result[0]: conn.close(); return jsonify({'error': 'Usuario sin RUT para actualizar'}), 404
        
        rut_path = result[0]
        # --- CORRECCIÓN DE RUTA ABSOLUTA ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_rut_path = os.path.join(script_dir, rut_path)
        doc = fitz.open(absolute_rut_path)
        # --- FIN DE CORRECCIÓN ---

        pdf_field_name = PDF_FIELD_MAP.get(field_to_update)
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
        
        # --- SOLUCIÓN DE GUARDADO ROBUSTO ---
        temp_path = absolute_rut_path + ".tmp"
        doc.save(temp_path, garbage=4, deflate=True)
        doc.close()
        os.remove(absolute_rut_path) # Borra el original
        os.rename(temp_path, absolute_rut_path) # Renombra el nuevo al nombre final
        # --- FIN DE LA SOLUCIÓN ---
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("UPDATE usuarios SET fecha_modificacion = ? WHERE cedula = ?", (current_time, user_cedula)); conn.commit(); conn.close()
        
        print(f"RUT de {user_name} actualizado. Campo: {field_to_update}"); return jsonify({'status': 'success', 'message': 'RUT actualizado.'})
        
    except Exception as e: 
        print(f"Error actualizando RUT: {e}"); return jsonify({'error': str(e)}), 500
# --- 5. RUTAS DE CHAT Y CALIFICACIÓN ---
@app.route('/chat', methods=['POST'])
def handle_chat_final():
    if 'user_id' not in session: return jsonify({'error': 'No hay sesión activa'}), 401
    if not model_gemini: return jsonify({'error': 'Modelo IA no configurado'}), 500
    prompt = request.json.get("prompt")
    if not prompt: return jsonify({'error': 'No se recibió pregunta'}), 400
    if 'chat_history' not in session: session['chat_history'] = []
    sentiment = "neutro"
    try:
        sentiment_prompt = f'Analiza el sentimiento. Responde solo con: positivo, negativo, neutro. Texto: "{prompt}"'
        sentiment_response = model_gemini.generate_content(sentiment_prompt)
        detected_sentiment = sentiment_response.text.strip().lower()
        if detected_sentiment in ['positivo', 'negativo', 'neutro']: sentiment = detected_sentiment
    except Exception as e: print(f"Error al analizar sentimiento: {e}")
    history_context = "\n".join([f"Usuario: {h['prompt']}\nAsistente: {h['response']}" for h in session['chat_history'][-3:]])
    specialized_prompt = f"""
    Eres AiVi, un asistente experto de la DIAN. Eres amable, cortés y muy empático.
    **Análisis de la Conversación:**
    - Sentimiento del último mensaje del usuario: **{sentiment}**
    **TUS INSTRUCCIONES DE COMPORTAMIENTO:**
    1.  ADAPTA TU TONO: Si el sentimiento es negativo, sé empático. Si es positivo, sé entusiasta. si es neutro manténte profesional.
    2.  USA TU BASE DE CONOCIMIENTO: Responde solo con el siguiente documento.
    3.  SÉ PROACTIVO: Si no sabes la respuesta, ofrece ayuda con otros trámites.
    --- DOCUMENTO DE CONOCIMIEN TO ---
    {KNOWLEDGE_BASE}
    --- FIN DEL DOCUMENTO ---
    Historial: {history_context}
    Responde a: {prompt}
    """
    try:
        response = model_gemini.generate_content(specialized_prompt)
        response_text = response.text
        session['chat_history'].append({"prompt": prompt, "response": response_text, "sentiment": sentiment})
        session.modified = True
        return jsonify({'response': markdown.markdown(response_text)})
    except Exception as e:
        print(f"Error en Gemini: {e}"); return jsonify({'error': f'Error al contactar al modelo de IA: {e}'}), 500
@app.route('/rate_chat', methods=['POST'])
def rate_chat():
    if 'user_id' not in session: return jsonify({'error': 'No hay sesión activa'}), 401
    rating = request.json.get('rating')
    user_cedula = session['user_id']
    user_name = session['user_name']
    history = session.get('chat_history', [])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
    cursor.execute("INSERT INTO calificaciones (user_cedula, rating, timestamp) VALUES (?, ?, ?)", (user_cedula, rating, timestamp))
    cursor.execute("SELECT email FROM usuarios WHERE cedula = ?", (user_cedula,))
    result = cursor.fetchone()
    user_email = result[0] if result else None
    conn.commit(); conn.close()
    print(f"Calificación: {rating} para {user_name}")
    if user_email and history:
        try:
            email_html = f"<h3>Hola {user_name},</h3><p>Gracias por usar el asistente AiVi. Aquí tienes una copia de tu conversación:</p><hr>"
            for item in history:
                email_html += f"<p><b>Tú:</b> {item['prompt']}</p>"
                email_html += f"<div><b>AiVi:</b> {markdown.markdown(item['response'])}</div><br>"
            msg = Message(subject="Transcripción de tu consulta con AiVi DIAN", sender=("Asistente AiVi", app.config['MAIL_USERNAME']), recipients=[user_email], html=email_html)
            mail.send(msg)
            print(f"Correo de transcripción enviado a {user_email}")
        except Exception as e:
            print(f"Error al enviar el correo: {e}")
    session.pop('chat_history', None)
    return jsonify({'status': 'success'})

# --- 6. RUTA PRINCIPAL Y ARRANQUE ---
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    setup_database()
    load_known_users()
    print(f"Servidor AiVi DIAN listo.")
    app.run(host='0.0.0.0', port=5000, debug=True)