import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Keras Model I trained on  Google Colab
model = tf.keras.models.load_model("chest_xray_model.h5")

# Labels
class_labels = ['NORMAL', 'PNEUMONIA']

# Preprocessing to increase the speed
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_only(image):
    img = Image.fromarray(image)
    preprocessed = preprocess_image(img)

    prediction = model.predict(preprocessed)[0]
    print("Raw prediction:", prediction)

    # for sigmoid
    if len(prediction) == 1:
        predicted_label = class_labels[int(prediction[0] > 0.5)]
    else:
        predicted_label = class_labels[np.argmax(prediction)]

    return f"Prediction: {predicted_label}"

# UI part
iface = gr.Interface(
    fn=predict_only,
    inputs=gr.Image(type="numpy", label="Upload Chest X-ray"),
    outputs=gr.Textbox(label="Model Prediction"),
    title="Chest X-ray Classifier",
    description="Upload a chest X-ray image. The model predicts if it is NORMAL or shows signs of PNEUMONIA."
)

# Launch the app
iface.launch()
