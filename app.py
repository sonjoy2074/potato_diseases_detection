import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
MODEL = tf.keras.models.load_model("../saved_model/potato_diseases_modelv1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def predict_image(img):
    # Ensure the image is resized to match model input
    img = img.resize((256, 256))
    img_array = np.expand_dims(np.array(img), 0)  # shape: (1, 256, 256, 3)
    predictions = MODEL.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return img, predicted_class, f"{confidence:.2f}"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload a potato leaf image"),
    outputs=[
        gr.Image(type="pil", label="Uploaded Image"),
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
    ],
    title="Potato Disease Classifier 🥔",
    description="Upload an image of a potato leaf to detect Early Blight, Late Blight, or Healthy."
)

if __name__ == "__main__":
    demo.launch()
