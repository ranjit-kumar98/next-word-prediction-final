import numpy as np

def predict_next_words(
    model,
    seed_text,
    word_to_index,
    index_to_word,
    sequence_length,
    top_k=5,
    temperature=1.0
):
    words = seed_text.lower().split()
    words = words[-sequence_length:]

    encoded = [word_to_index.get(w, 0) for w in words]
    if len(encoded) < sequence_length:
        encoded = [0] * (sequence_length - len(encoded)) + encoded

    encoded = np.array(encoded).reshape(1, -1)
    preds = model.predict(encoded, verbose=0)[0]

    preds = np.log(preds + 1e-9) / temperature
    preds = np.exp(preds) / np.sum(np.exp(preds))

    top_indices = preds.argsort()[-top_k:][::-1]

    return [
        {"word": index_to_word[i], "probability": float(preds[i])}
        for i in top_indices
    ]
