import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile

# Load model
model = tf.keras.models.load_model("saved_model.h5", compile=False)

# ---------------- PREDICTION ----------------
def predict(img):
    img = img.resize((128, 128))
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]

    fake = float(prediction[0]) * 100
    real = float(prediction[1]) * 100

    label = "FAKE" if fake > real else "REAL"
    confidence = max(fake, real)

    return label, confidence, fake, real

# ---------------- BAR CHART ----------------
def plot_chart(fake, real):
    plt.figure()
    plt.bar(["FAKE", "REAL"], [fake, real])
    plt.ylabel("Confidence %")
    plt.title("Prediction Confidence")
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    plt.savefig(path)
    plt.close()
    return path

# ---------------- PDF REPORT ----------------
def generate_pdf(label, confidence, fake, real):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    pdf.cell(200, 10, txt="AI Image Detection Report", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {label}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"FAKE Score: {fake:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"REAL Score: {real:.2f}%", ln=True)

    path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    pdf.output(path)
    return path

# ---------------- MAIN FUNCTION ----------------
def analyze(img):
    label, confidence, fake, real = predict(img)

    chart = plot_chart(fake, real)
    report = generate_pdf(label, confidence, fake, real)

    return img, label, f"{confidence:.2f}%", chart, report

# ---------------- SAMPLE IMAGES (DEMO MODE) ----------------
def load_sample():
    return None

# ---------------- UI ----------------
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")

with gr.Blocks(theme=theme) as demo:

    gr.Markdown("""
    # 🧠 AI Image Detection System (CIFAKE)
    ### Detect whether an image is REAL or AI-generated
    """)

    with gr.Row():

        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload / Try Sample Image")
            btn = gr.Button("🚀 Analyze")

        with gr.Column():
            img_out = gr.Image(label="Preview")
            label_out = gr.Textbox(label="Prediction")
            conf_out = gr.Textbox(label="Confidence")

    with gr.Row():
        chart_out = gr.Image(label="📊 Confidence Chart")
        pdf_out = gr.File(label="📄 Download Report")

    btn.click(
        fn=analyze,
        inputs=img_input,
        outputs=[img_out, label_out, conf_out, chart_out, pdf_out]
    )

demo.launch()
