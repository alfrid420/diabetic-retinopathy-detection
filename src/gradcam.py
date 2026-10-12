"""Grad-CAM helpers for the retinopathy classifier (MobileNetV2 backbone)."""
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (160, 160)
LAST_CONV_LAYER = "Conv_1"  # last conv layer of MobileNetV2 before pooling


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER, pred_index=None):
    """Return (heatmap in [0, 1], predicted class index) for a batch of one image."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4):
    """Blend a Grad-CAM heatmap onto a PIL image (already resized to IMG_SIZE)."""
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return np.uint8(np.array(img) * (1 - alpha) + color * alpha)


def predict_with_gradcam(img: Image.Image, model):
    """Predict the category for one PIL image and return probs + Grad-CAM overlay."""
    img = img.convert("RGB").resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    probs = model.predict(img_array, verbose=0)[0]
    pred_class = int(np.argmax(probs))

    heatmap, _ = make_gradcam_heatmap(img_array, model, pred_index=pred_class)
    overlay = overlay_heatmap(img, heatmap)
    return pred_class, probs, np.array(img), overlay
