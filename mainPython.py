import json
import numpy as np
import random
import spacy
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout




# Cargar modelo spaCy
nlp = spacy.load('en_core_web_sm')

# Función para cargar datos desde JSON
def cargar_datos(file_path='intents.json'):
    with open(file_path, encoding='utf-8') as file:
        data = json.load(file)
    return data

# Función para guardar datos en el archivo JSON
def guardar_datos(data, data_file='intents.json'):
    with open(data_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)

# Preprocesamiento de datos con spaCy
def preprocesar_datos(data):
    words = []
    classes = []
    documents = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            doc = nlp(pattern)
            word_list = [token.lemma_.lower() for token in doc if not token.is_punct]
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = sorted(set(words))
    classes = sorted(set(classes))
    return words, classes, documents

def preparar_entrenamiento(words, classes, documents):
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = [1 if w in doc[0] else 0 for w in words]
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array([np.pad(bag, (0, len(words) - len(bag))) for bag in training[:, 0]])  # Asegurar que todas las bolsas de palabras tengan la misma longitud
    train_y = np.array(training[:, 1].tolist())
    
    return train_x, train_y

def crear_modelo(input_shape, output_shape):
    model = Sequential([
        Dense(256, input_shape=(input_shape,), activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_shape, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def entrenar_modelo(model, train_x, train_y, epochs=500, batch_size=8):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])
    return model

def guardar_modelo(model, file_path='chatbot_model.keras'):
    model.save(file_path)

def cargar_modelo(file_path='chatbot_model.keras'):
    return load_model(file_path)

def clean_up_sentence(sentence):
    doc = nlp(sentence)
    return [token.lemma_.lower() for token in doc if not token.is_punct]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.pad(bag, (0, len(words) - len(bag)))  # Asegurar que la bolsa de palabras tenga la longitud correcta

def predict_class(sentence, model, words, classes):
    p = bow(sentence, words)
    p = np.pad(p, (0, len(words) - len(p)))  # Asegurarse de que la entrada tenga la longitud correcta
    res = model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def add_new_intent(user_input, user_response):
    data = cargar_datos()
    new_intent = {
        "tag": user_input,
        "patterns": [user_input],
        "responses": [user_response]
    }
    data['intents'].append(new_intent)
    guardar_datos(data)
    return data

def retrain_model():
    data = cargar_datos()
    words, classes, documents = preprocesar_datos(data)
    train_x, train_y = preparar_entrenamiento(words, classes, documents)

    model = crear_modelo(len(train_x[0]), len(train_y[0]))
    model = entrenar_modelo(model, train_x, train_y)
    guardar_modelo(model)
    return words, classes, model

# Inicializar datos y modelo
data = cargar_datos()
words, classes, documents = preprocesar_datos(data)
train_x, train_y = preparar_entrenamiento(words, classes, documents)

try:
    model = cargar_modelo()
    print("Modelo cargado desde el archivo.")
except:
    print("No se encontró un modelo existente. Entrenando un nuevo modelo.")
    model = crear_modelo(len(train_x[0]), len(train_y[0]))
    model = entrenar_modelo(model, train_x, train_y)
    guardar_modelo(model)

print("¡Chatbot listo para hablar contigo!")

# # Reentrenar el modelo después de los cambios
# model = crear_modelo(len(train_x[0]), len(train_y[0]))
# model = entrenar_modelo(model, train_x, train_y)
# guardar_modelo(model)

while True:
    message = input("Tú: ")
    ints = predict_class(message, model, words, classes)

    if len(ints) > 0:  # Verificar que la lista ints no esté vacía
        if ints[0]['intent'] == "despedida" or ints[0]['intent'] == "despedida_fluida":
            res = get_response(ints, data)
            print(f'Tú: {message}')
            print(f'Chatbot: {res}')
            break  # Termina el bucle y la conversación

        res = get_response(ints, data)
        print(f'Tú: {message}')
        print(f'Chatbot: {res}')

        is_correct = input("¿La respuesta del chatbot fue correcta? (sí/no): ").strip().lower()
        if is_correct == "no":
            user_response = input("Por favor ingresa la respuesta correcta del chatbot: ")
            data = add_new_intent(message, user_response)
            words, classes, model = retrain_model()
        else:
            print("¡Genial! Continuemos.")
    else:
        print("No se pudo clasificar el mensaje. Por favor, intenta de nuevo.")

