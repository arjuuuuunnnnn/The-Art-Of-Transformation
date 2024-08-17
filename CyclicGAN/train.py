import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import ImageSketchDataset, Generator, Discriminator, weights_init, train_step

# Fixed hyperparameters
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 2e-4

# Fixed paths for Kaggle
IMAGE_DIR = '/kaggle/input/cuhk-face-sketch-database-cufs/photos'
SKETCH_DIR = '/kaggle/input/cuhk-face-sketch-database-cufs/sketches'
RESULTS_DIR = '/kaggle/working/results'
MODEL_SAVE_DIR = '/kaggle/working/models'

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageSketchDataset(image_dir=IMAGE_DIR,
                             sketch_dir=SKETCH_DIR,
                             transform=transform)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator_g = Generator().to(device)
generator_f = Generator().to(device)
discriminator_x = Discriminator().to(device)
discriminator_y = Discriminator().to(device)

generator_g.apply(weights_init)
generator_f.apply(weights_init)
discriminator_x.apply(weights_init)
discriminator_y.apply(weights_init)

optimizer_g_g = optim.Adam(generator_g.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_g_f = optim.Adam(generator_f.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_d_x = optim.Adam(discriminator_x.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_d_y = optim.Adam(discriminator_y.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

scheduler_g_g = StepLR(optimizer_g_g, step_size=10, gamma=0.1)
scheduler_g_f = StepLR(optimizer_g_f, step_size=10, gamma=0.1)
scheduler_d_x = StepLR(optimizer_d_x, step_size=10, gamma=0.1)
scheduler_d_y = StepLR(optimizer_d_y, step_size=10, gamma=0.1)

def train(dataloader, epochs):
    for epoch in range(epochs):
        for i, (real_x, real_y) in enumerate(dataloader):
            loss_g, loss_d_x, loss_d_y = train_step(real_x, real_y)
            if i % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Step {i}, Loss G: {loss_g:.4f}, Loss D_X: {loss_d_x:.4f}, Loss D_Y: {loss_d_y:.4f}')

        scheduler_g_g.step()
        scheduler_g_f.step()
        scheduler_d_x.step()
        scheduler_d_y.step()

        # Save models periodically or after training
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            torch.save(generator_g.state_dict(), os.path.join(MODEL_SAVE_DIR, f'generator_g_epoch_{epoch+1}.pth'))
            torch.save(generator_f.state_dict(), os.path.join(MODEL_SAVE_DIR, f'generator_f_epoch_{epoch+1}.pth'))
            torch.save(discriminator_x.state_dict(), os.path.join(MODEL_SAVE_DIR, f'discriminator_x_epoch_{epoch+1}.pth'))
            torch.save(discriminator_y.state_dict(), os.path.join(MODEL_SAVE_DIR, f'discriminator_y_epoch_{epoch+1}.pth'))

            print(f'Models saved after epoch {epoch+1}')

if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Starting training for {EPOCHS} epochs")
    train(dataloader, epochs=EPOCHS)
    print("Training completed")
