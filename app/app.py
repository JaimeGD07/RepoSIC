from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import cv2
import numpy as np
import base64
import os

# Configuración inicial de la aplicación
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')  # Lee la clave
CORS(app)

# Configuración de rutas importantes
app.config.update({
    'MODEL_PATH': os.path.join(app.root_path, 'data', 'modelo_random_forest.pkl'),
    'IMG_SIZE': (512, 384)
})

# Cargar el modelo de ML al iniciar
try:
    with open(app.config['MODEL_PATH'], "rb") as model_file:
        clf = pickle.load(model_file)
    print("✅ Modelo cargado correctamente desde:", app.config['MODEL_PATH'])
except Exception as e:
    print(f"❌ Error cargando el modelo: {str(e)}")
    clf = None

def base64_to_image(base64_str):
    """Convierte una cadena base64 a imagen OpenCV"""
    try:
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error procesando imagen: {str(e)}")
        return None

def predecir_imagen(img_base64):
    """Realiza la predicción con el modelo"""
    if not clf:
        return None
        
    img = base64_to_image(img_base64)
    if img is None:
        return None
        
    # Preprocesamiento
    img = cv2.resize(img, app.config['IMG_SIZE'])
    img = img / 255.0  # Normalización
    img_flat = img.flatten().reshape(1, -1)
    
    return clf.predict(img_flat)[0]

# Rutas de la aplicación
@app.route('/')
def home():
    """Sirve la página principal"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para predicciones"""
    if not clf:
        return jsonify({'error': 'Modelo no disponible'}), 500
        
    try:
        data = request.json
        img_base64 = data['image']
        
        categoria = {
            0: "nulo",
            1: "carton",
            2: "papel",
            3: "metal"
        }
        
        prediccion = predecir_imagen(img_base64)
        if prediccion is None:
            return jsonify({'error': 'Error procesando imagen'}), 400
            
        return jsonify({
            'categoria': categoria.get(prediccion + 1, 'desconocido'),
            'codigo': int(prediccion + 1)
        })
        
    except Exception as e:
        print(f"Error en /predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Configuración para desarrollo
    app.run(host='0.0.0.0', port=5000, debug=True)