import os

class Config:
    # Configuración general
    SECRET_KEY = os.getenv('SECRET_KEY')
    IMG_SIZE = (512, 384)
    
    # Rutas de los modelos
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    MODEL_IMAGE_PATH = os.path.join(BASE_DIR, 'data', 'modelo_random_forest.pkl')
    MODEL_CHATBOT_PATH = os.path.join(BASE_DIR, 'data', 'chatbot_model.pkl')
    VECTORIZER_PATH = os.path.join(BASE_DIR, 'data', 'vectorizer.pkl')