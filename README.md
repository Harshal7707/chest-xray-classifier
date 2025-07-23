# Chest X-ray Classifier

This project is a simple AI-powered web app that classifies chest X-ray images as either **NORMAL** or showing signs of **PNEUMONIA**.

---


<img width="1918" height="791" alt="image" src="https://github.com/user-attachments/assets/d7515676-99ac-4948-944b-dd58f3d07179" />

## Overview

- The model was **trained on Google Colab** using a publicly available chest X-ray dataset.
- After training, the Keras model was saved as an `.h5` file and used here for inference.
- This app loads the trained model and provides an easy-to-use **Gradio** interface for image upload and classification.
- The app predicts the class and shows the result instantly.

---

## Gemini Integration (Concept / Initial Plan)

- Initially, the project was designed to use **Google Gemini** API to generate detailed, AI-powered medical diagnostic reports based on the model's prediction.
- Gemini was intended to take the prediction label (e.g., "PNEUMONIA") and generate a human-readable report including symptoms, treatment suggestions, and follow-ups.
- Due to current access limitations with Gemini, the project was adapted to use other alternatives like Hugging Face models for generating text content.
- This integration showcases how combining image classification with advanced language models can provide an end-to-end AI-assisted diagnostic tool.

---

## Features

- Accepts chest X-ray images as input.
- Preprocesses images to the model's expected size and format.
- Predicts whether the X-ray is **NORMAL** or has **PNEUMONIA**.
- Simple, user-friendly web interface built with Gradio.
- (Planned) AI-generated detailed medical reports powered by Gemini or similar large language models.

---

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/Harshal7707/chest-xray-classifier.git
   cd chest-xray-classifier
