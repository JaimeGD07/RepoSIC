import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '../data/modelo_random_forest.pkl')
    IMG_SIZE = (512, 384)