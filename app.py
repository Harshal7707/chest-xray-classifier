import gradio as gr
import tensorflow as tf
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = tf.keras.models.load_model("chest_xray_model.h5")

class_labels = ['NORMAL', 'PNEUMONIA']

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def generate_report(label):
    prompt = f"""Generate a detailed medical-style diagnostic report for a chest X-ray image predicted to show: {label}.
Include likely symptoms, treatment suggestions, and cautionary follow-ups in simple language."""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

def predict_and_generate(image):
    img = Image.fromarray(image)
    preprocessed = preprocess_image(img)
    prediction = model.predict(preprocessed)[0]
    predicted_label = class_labels[np.argmax(prediction)]
    report = generate_report(predicted_label)
    return f"Prediction: {predicted_label}", report

iface = gr.Interface(
    fn=predict_and_generate,
    inputs=gr.Image(type="numpy", label="Upload Chest X-ray"),
    outputs=[
        gr.Textbox(label="Model Prediction"),
        gr.Textbox(label="AI-Generated Medical Report")
    ],
    title="Chest X-ray Report Generator (Gemini-powered)",
    description="Upload a chest X-ray image. The model predicts and Gemini generates a report."
)

iface.launch()
