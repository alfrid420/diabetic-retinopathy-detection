"""Gradio demo: upload a retina scan, get the predicted category + Grad-CAM heatmap.

Usage:
    python app.py                      # expects models/retinopathy_model.keras
    python app.py --model path/to.keras
"""
import argparse
import sys

import gradio as gr
import tensorflow as tf

sys.path.insert(0, "src")
from gradcam import predict_with_gradcam  # noqa: E402

# Dataset labels are plain categories 1-3. Rename here only if the client confirms
# the clinical meaning of each category.
CLASS_NAMES = ["Category 1", "Category 2", "Category 3"]


def build_demo(model):
    def predict(image):
        pred, probs, _, overlay = predict_with_gradcam(image, model)
        label = f"{CLASS_NAMES[pred]} ({probs[pred] * 100:.1f}% confidence)"
        prob_dict = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
        return label, prob_dict, overlay

    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload a Retina Scan"),
        outputs=[
            gr.Text(label="Predicted Category"),
            gr.Label(label="Class Probabilities"),
            gr.Image(label="Grad-CAM Heatmap"),
        ],
        title="Diabetic Retinopathy Category Detector",
        description=(
            "Upload a retina fundus scan to predict its retinopathy category, with a "
            "Grad-CAM heatmap showing which regions influenced the prediction. "
            "Educational demo only - not a medical device."
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/retinopathy_model.keras")
    args = parser.parse_args()
    build_demo(tf.keras.models.load_model(args.model)).launch()
