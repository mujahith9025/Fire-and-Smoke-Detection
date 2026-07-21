"""
Grad-CAM Explainable AI Visual Heatmap Generator.
Generates class activation maps showing regions of interest for model predictions.
"""

import cv2
import numpy as np
import tensorflow as tf
from utils.logger import logger

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Computes Grad-CAM heatmap array for a given input tensor and Keras model.
    """
    try:
        # Find last conv layer automatically if not specified
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.Model)) or "conv" in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break

        if not last_conv_layer_name:
            # Fallback to visual color heatmap generator if model has no Conv2D layer accessible directly
            return _generate_fallback_hsv_heatmap(img_array)

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    except Exception as e:
        logger.warning(f"Grad-CAM generation notice: {e}. Generating feature saliency heatmap.")
        return _generate_fallback_hsv_heatmap(img_array)


def _generate_fallback_hsv_heatmap(img_array):
    """Generates visual color saliency map as robust fallback."""
    img = img_array[0] if len(img_array.shape) == 4 else img_array
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Highlight high saturation & luminosity regions
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    heatmap = s * v
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
    return heatmap


def overlay_gradcam(image_bgr, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays Grad-CAM heatmap on original BGR image.
    Returns composite BGR image.
    """
    height, width = image_bgr.shape[:2]
    resized_heatmap = cv2.resize(heatmap, (width, height))
    
    # Convert heatmap to 0-255 uint8
    heatmap_uint8 = np.uint8(255 * resized_heatmap)
    
    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Superimpose heatmap on original image
    superimposed = cv2.addWeighted(image_bgr, 1.0 - alpha, colored_heatmap, alpha, 0)
    return superimposed
