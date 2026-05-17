import os
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def run_gradcam(model, image_tensor, target_layer) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single image.

    Args:
        model: The EfficientNet-B3 detector (nn.Module).
        image_tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
        target_layer: The convolutional layer to visualise
                      (e.g. model.conv_head for EfficientNet-B3).

    Returns:
        Heatmap as a numpy array of shape (224, 224) with values in [0, 1].
    """
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor)
    return grayscale_cam[0, :]


def save_heatmap(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    save_path: str,
) -> None:
    """Overlay a Grad-CAM heatmap on the original image and save.

    Args:
        heatmap: Array of shape (H, W) with values in [0, 1].
        original_image: RGB image as numpy array with values in [0, 1],
                        shape (H, W, 3).
        save_path: Destination file path (e.g. 'outputs/heatmaps/sample.png').
    """
    overlay = show_cam_on_image(original_image, heatmap, use_rgb=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
