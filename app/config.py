import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '../data/modelo_random_forest.pkl')
    IMG_SIZE = (512, 384)

import os

class Config:
    # Configuración general
    SECRET_KEY = os.getenv('SECRET_KEY')  # Valor por defecto para desarrollo
    IMG_SIZE = (512, 384)
    
    # Rutas de los modelos (ajustadas a tu estructura de carpetas)
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Raíz del proyecto (donde está `app/`)
    MODEL_IMAGE_PATH = os.path.join(BASE_DIR, 'data', 'modelo_random_forest.pkl')
    MODEL_CHATBOT_PATH = os.path.join(BASE_DIR, 'data', 'chatbot_model.pkl')
    VECTORIZER_PATH = os.path.join(BASE_DIR, 'data', 'vectorizer.pkl')