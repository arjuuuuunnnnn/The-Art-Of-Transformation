import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class ImageSketchDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, sketch_dir, transform=None):
        self.image_dir = image_dir
        self.sketch_dir = sketch_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.sketch_files = sorted(os.listdir(sketch_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        sketch_path = os.path.join(self.sketch_dir, self.sketch_files[idx])
        image = Image.open(image_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            sketch = self.transform(sketch)

        return image, sketch

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(Generator, self).__init__()

        model = [
            nn.Conv2d(3, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.Conv2d(64, 3, 7, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_step(real_x, real_y, generator_g, generator_f, discriminator_x, discriminator_y, 
               optimizer_g_g, optimizer_g_f, optimizer_d_x, optimizer_d_y, 
               criterion_GAN, criterion_cycle, criterion_identity, device):
    real_x = real_x.to(device)
    real_y = real_y.to(device)

    optimizer_g_g.zero_grad()
    optimizer_g_f.zero_grad()

    same_y = generator_g(real_y)
    loss_identity_y = criterion_identity(same_y, real_y)
    same_x = generator_f(real_x)
    loss_identity_x = criterion_identity(same_x, real_x)

    fake_y = generator_g(real_x)
    pred_fake_y = discriminator_y(fake_y)
    loss_gan_g = criterion_GAN(pred_fake_y, torch.ones_like(pred_fake_y))

    fake_x = generator_f(real_y)
    pred_fake_x = discriminator_x(fake_x)
    loss_gan_f = criterion_GAN(pred_fake_x, torch.ones_like(pred_fake_x))

    recovered_x = generator_f(fake_y)
    loss_cycle_x = criterion_cycle(recovered_x, real_x)

    recovered_y = generator_g(fake_x)
    loss_cycle_y = criterion_cycle(recovered_y, real_y)

    loss_g = loss_gan_g + loss_gan_f + (loss_cycle_x + loss_cycle_y) * 10 + (loss_identity_x + loss_identity_y) * 5

    loss_g.backward()
    optimizer_g_g.step()
    optimizer_g_f.step()

    optimizer_d_x.zero_grad()

    pred_real_x = discriminator_x(real_x)
    loss_real_x = criterion_GAN(pred_real_x, torch.ones_like(pred_real_x))

    pred_fake_x = discriminator_x(fake_x.detach())
    loss_fake_x = criterion_GAN(pred_fake_x, torch.zeros_like(pred_fake_x))

    loss_d_x = (loss_real_x + loss_fake_x) * 0.5
    loss_d_x.backward()
    optimizer_d_x.step()

    optimizer_d_y.zero_grad()

    pred_real_y = discriminator_y(real_y)
    loss_real_y = criterion_GAN(pred_real_y, torch.ones_like(pred_real_y))

    pred_fake_y = discriminator_y(fake_y.detach())
    loss_fake_y = criterion_GAN(pred_fake_y, torch.zeros_like(pred_fake_y))

    loss_d_y = (loss_real_y + loss_fake_y) * 0.5
    loss_d_y.backward()
    optimizer_d_y.step()

    return loss_g.item(), loss_d_x.item(), loss_d_y.item()

def generate_and_save_images(generator, test_input, test_sketch, epoch, filename, device, results_dir):
    generator.eval()
    with torch.no_grad():
        prediction = generator(test_input.to(device))
    generator.train()

    test_sketch = test_sketch[0].cpu().permute(1, 2, 0) * 0.5 + 0.5
    prediction = prediction[0].cpu().permute(1, 2, 0) * 0.5 + 0.5

    plt.figure(figsize=(5, 5))
    plt.imshow(prediction)
    plt.axis('off')
    plt.savefig(os.path.join(results_dir, f'{filename}_generated_epoch_{epoch}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_sketch)
    plt.title('Real Sketch')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(prediction)
    plt.title('Generated Sketch')
    plt.axis('off')

    plt.savefig(os.path.join(results_dir, f'{filename}_comparison_epoch_{epoch}.png'))
    plt.close()
