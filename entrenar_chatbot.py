# train_chatbot.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Datos de entrenamiento
data = [
    ("hola", "saludo"),
    ("¿Cómo estás?", "saludo"),
    ("Buenos días", "saludo"),
    ("Buenas tardes", "saludo"),
    ("Buenas noches", "saludo"),
    ("¿Qué tal?", "saludo"),
    ("¿Cómo te va?", "saludo"),
    ("¡Qué onda!", "saludo"),
    ("¡Hola, qué tal!", "saludo"),
    ("Adiós", "despedida"),
    ("Nos vemos", "despedida"),
    ("Hasta luego", "despedida"),
    ("Hasta pronto", "despedida"),
    ("Chao", "despedida"),
    ("Cuídate", "despedida"),
    ("Nos vemos pronto", "despedida"),
    ("Hasta mañana", "despedida"),
    ("Que tengas un buen día", "despedida"),
    ("¡Adiós, cuídate!", "despedida"),
    ("¿Quién eres?", "identidad"),
    ("¿Eres un humano o un robot?", "identidad"),
    ("¿Cómo puedes ayudarme con el reciclaje?", "identidad"),
    ("¿Cuál es tu propósito?", "identidad"),
    ("¿Por qué fuiste creado?", "identidad"),
    ("¿Sabes sobre el medio ambiente?", "identidad"),
    ("¿Eres un asistente ecológico?", "identidad"),
    ("¿Tienes un nombre?", "identidad"),
    ("¿Por qué promueves el reciclaje?", "identidad"),
    ("¿Cómo aprendiste sobre reciclaje?", "identidad"),
    ("¿Qué puedo reciclar?", "quereciclar"),
    ("¿Dónde están las zonas de reciclaje en El Salvador?", "reciclaje"),
    ("¿Qué materiales puedo reciclar?", "quereciclar"),
    ("¿Por qué es importante reciclar?", "importanciareciclaje"),
    ("¿Cómo puedo empezar a reciclar?", "comoreciclar"),
    ("¿Qué es la economía circular?", "circular"),


]

# Separar los mensajes y las etiquetas
X, y = zip(*data)

# Vectorizar los mensajes
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Entrenar el modelo
model = MultinomialNB()
model.fit(X_vectorized, y)

# Guardar el modelo y el vectorizador
with open("app/chatbot_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("app/vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Modelo y vectorizador guardados.")
