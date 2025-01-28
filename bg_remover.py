import gradio as gr
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from u2net import U2NET  # Import model architecture

# Load the U^2-Net model
model_path = "u2net.pth"  # Make sure this file exists in the same directory
model = U2NET(3, 1)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Preprocessing function
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Increase resolution for better detail
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

# Postprocessing function
def postprocess(output, original_size):
    output = output.squeeze().detach().numpy()
    output = (output - output.min()) / (output.max() - output.min())  # Normalize to [0, 1]
    mask = (output * 255).astype(np.uint8)  # Scale to [0, 255]

    # Resize to match the original image size
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)

    # Apply threshold to sharpen the mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Optional: Use morphological operations to smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


# Gradio interface function
def remove_background(image):
    original_size = image.size
    input_tensor = preprocess(image)
    with torch.no_grad():
        output = model(input_tensor)[0]  # Get output from the model
    mask = postprocess(output, original_size)

    # Create RGBA image
    transparent_img = np.array(image).astype(np.uint8)
    if transparent_img.shape[2] == 3:  # If RGB, convert to RGBA
        transparent_img = cv2.cvtColor(transparent_img, cv2.COLOR_RGB2RGBA)

    # Feather the mask for smoother edges
    alpha = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0  # Normalize to [0, 1]
    for c in range(3):  # Apply alpha blending to RGB channels
        transparent_img[..., c] = transparent_img[..., c] * alpha
    transparent_img[..., 3] = (mask > 0).astype(np.uint8) * 255  # Keep alpha binary

    return Image.fromarray(transparent_img, "RGBA")

# Gradio app
iface = gr.Interface(
    fn=remove_background,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Background Remover",
    description="Upload an image to remove its background using U^2-Net."
)

if __name__ == "__main__":
    iface.launch()
