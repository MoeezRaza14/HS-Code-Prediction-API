from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load your model, vectorizer, and label encoder
model = tf.keras.models.load_model("model.h5", compile=False)
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define a request model
class DescriptionRequest(BaseModel):
    descriptions: list[str]

@app.post("/predict")
async def predict(data: DescriptionRequest):
    # Vectorize each description in the list
    vectorized_texts = vectorizer.transform(data.descriptions)

    # Make predictions for each vectorized text
    predictions = model.predict(vectorized_texts.toarray())

    # For each product, get the top 3 predictions and their probabilities
    results = []
    for i, prediction in enumerate(predictions):
        top_3_indices = np.argsort(prediction)[-3:][::-1]  # Get indices of top 3 probabilities
        top_3_labels = label_encoder.inverse_transform(top_3_indices)
        top_3_probabilities = prediction[top_3_indices] * 100  # Convert to percentage
        prods = data.descriptions[i]

        # Combine labels and their probabilities into a dictionary
        hs_codes_probabilities = {
            label: f"{probability:.2f}%"
            for label, probability in zip(top_3_labels, top_3_probabilities)
        }

        result = {
            "Product": prods,
            "HS Codes & Probabilities": hs_codes_probabilities
        }
        results.append(result)

    return {"predictions": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
