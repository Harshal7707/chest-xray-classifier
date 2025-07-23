def generate_report(label):
    prompt = f"""Generate a detailed medical-style diagnostic report for a chest X-ray image predicted to show: {label}.
Include likely symptoms, treatment suggestions, and cautionary follow-ups in simple language."""

    model = genai.GenerativeModel("models/gemini-pro")
    response = model.generate_content(prompt)
    return response.text
