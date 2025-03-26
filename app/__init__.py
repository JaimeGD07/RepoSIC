from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    
    # Configuración básica
    app.config.from_pyfile('config.py')
    
    # Inicializar extensiones
    from flask_cors import CORS
    CORS(app)
    
    # Cargar modelo ML al iniciar
    with app.app_context():
        from app.services.ml_service import load_model
        app.clf = load_model(app.config['MODEL_PATH'])
    
    # Registrar rutas
    from app import routes
    app.register_blueprint(routes.bp)
    
    return app