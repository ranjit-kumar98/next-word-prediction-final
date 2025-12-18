import tensorflow as tf
import pickle

MODEL_PATH = "saved_model/language_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

with open("saved_model/word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)

with open("saved_model/index_to_word.pkl", "rb") as f:
    index_to_word = pickle.load(f)

SEQUENCE_LENGTH = 10
