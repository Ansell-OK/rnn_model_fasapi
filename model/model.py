import tensorflow as tf 
from pathlib import Path 

BASE_DIR = Path(__file__).resolve(strict=True).parent

model = tf.saved_model.load(f"{BASE_DIR}/dokinta_model/1")

class_names = [
    'Common Cold', 
    'Degue Fever', 
    'Malaria', 
    'Typhoid'
]

def predict_text(text):
    text_tensor = tf.constant([text], shape=(1,1), name='input_2')
    text_predict = model(text_tensor, training= False)
    prediction_change = tf.argmax(text_predict, axis = 1)

    prediction = class_names[prediction_change[0]]

    return prediction