from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    
    # Configuración básica
    app.config.from_pyfile('config.py')  # Si usas config.py
    
    # Inicializar extensiones (CORS, DB, etc.)
    from flask_cors import CORS
    CORS(app)
    
    # Cargar modelo ML al iniciar
    with app.app_context():
        from app.services.ml_service import load_model
        app.clf = load_model(app.config['MODEL_PATH'])
    
    # Registrar rutas
    from app import routes  # Importa tus rutas
    app.register_blueprint(routes.bp)  # Si usas Blueprints
    
    return app