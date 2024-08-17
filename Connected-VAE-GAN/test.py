import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

def deprocess_image(img):
    img = (img + 1.0) * 127.5  # Denormalize
    return np.clip(img, 0, 255).astype(np.uint8)

def test_single_image(gan, image_path):
    # Load and preprocess the test image
    test_image = preprocess_image(image_path)

    # Generate sketch using the generator
    generated_sketch = gan.generator(test_image, training=False)

    # Optionally, reconstruct the image from the sketch
    reconstructed_image = gan.generator(generated_sketch, training=False)

    # Deprocess images for display
    original_image = deprocess_image(test_image[0])
    generated_sketch = deprocess_image(generated_sketch[0])
    reconstructed_image = deprocess_image(reconstructed_image[0])

    # Display results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(generated_sketch, cmap='gray')
    axs[1].set_title('Generated Sketch')
    axs[1].axis('off')

    axs[2].imshow(reconstructed_image)
    axs[2].set_title('Reconstructed Image')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

# Usage
image_path = '/content/photos/f-005-01.jpg'
test_single_image(gan, image_path)
