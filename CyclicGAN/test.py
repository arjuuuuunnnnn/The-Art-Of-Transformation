import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import Generator

# Fixed paths
RESULTS_DIR = '/kaggle/working/results'
MODEL_PATH = '/kaggle/working/saved_model.pth'  # Update this path if your model is saved elsewhere

# Create directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the trained generator model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
generator.eval()

# Define transformation for input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def process_image(image_path):
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Generate the output
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # Convert tensors back to images
    input_image = input_tensor.squeeze(0).cpu().permute(1, 2, 0) * 0.5 + 0.5
    output_image = output_tensor.squeeze(0).cpu().permute(1, 2, 0) * 0.5 + 0.5

    return input_image, output_image

def display_and_save(input_image, output_image, image_name):
    # Display images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.title('Generated Image')
    plt.axis('off')

    # Save the results
    result_path = os.path.join(RESULTS_DIR, f'{image_name}_result.png')
    plt.savefig(result_path)
    plt.show()

    print(f"Results saved at: {result_path}")

if __name__ == "__main__":
    # Ask the user to provide an image path
    image_path = input("Please enter the path to your image: ")

    if os.path.exists(image_path):
        input_image, output_image = process_image(image_path)

        # Get image name without extension
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        display_and_save(input_image, output_image, image_name)
    else:
        print(f"File not found: {image_path}")
