import matplotlib.pyplot as plt
import numpy as np

def decode_segmap(mask):
    colors = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
    }
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in colors.items():
        rgb[mask == k] = v
    return rgb

def visualize_prediction(image, label, pred):
    image_np = image.cpu().numpy()
    if image_np.shape[0] > 1:
        image_np = image_np[0]  # Show first modality
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np, cmap='gray')
    plt.title("Input")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(decode_segmap(label.cpu().numpy()))
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(decode_segmap(pred.cpu().numpy()))
    plt.title("Prediction")
    plt.axis('off')

    plt.show()
