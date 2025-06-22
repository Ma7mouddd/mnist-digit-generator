import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 100
num_classes = 10
batch_size = 128
learning_rate = 0.0002
num_epochs = 50
beta1 = 0.5

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load MNIST dataset
mnist_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size=28):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        # Generator layers
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Combine noise and label embeddings
        label_embed = self.label_emb(labels)
        gen_input = torch.cat([noise, label_embed], dim=1)
        
        # Generate image
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size=28):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        
        # Discriminator layers
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size * 2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        
        # Get label embeddings
        label_embed = self.label_emb(labels)
        
        # Combine image and label
        disc_input = torch.cat([img_flat, label_embed], dim=1)
        
        # Discriminate
        validity = self.model(disc_input)
        return validity

# Initialize models
generator = Generator(latent_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Training function
def train_gan():
    generator.train()
    discriminator.train()
    
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            batch_size = imgs.shape[0]
            
            # Ground truth labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Real images and labels
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real images
            real_pred = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(real_pred, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_labels_gen = torch.randint(0, num_classes, (batch_size,)).to(device)
            fake_imgs = generator(noise, fake_labels_gen)
            fake_pred = discriminator(fake_imgs.detach(), fake_labels_gen)
            d_fake_loss = adversarial_loss(fake_pred, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake images
            fake_pred = discriminator(fake_imgs, fake_labels_gen)
            g_loss = adversarial_loss(fake_pred, real_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Print statistics
            if i % 200 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        
        # Save sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_sample_images(epoch + 1)
    
    # Save final models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    print("Training completed! Models saved.")

def save_sample_images(epoch):
    """Save sample generated images"""
    generator.eval()
    with torch.no_grad():
        # Generate one image for each digit
        noise = torch.randn(10, latent_dim).to(device)
        labels = torch.arange(0, 10).to(device)
        
        fake_imgs = generator(noise, labels)
        fake_imgs = fake_imgs.cpu().numpy()
        
        # Denormalize images
        fake_imgs = (fake_imgs + 1) / 2
        
        # Create subplot
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(10):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(fake_imgs[i, 0], cmap='gray')
            axes[row, col].set_title(f'Digit {i}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'sample_epoch_{epoch}.png')
        plt.close()
    
    generator.train()

def generate_digit_images(digit, num_images=5):
    """Generate images for a specific digit"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim).to(device)
        labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
        
        fake_imgs = generator(noise, labels)
        fake_imgs = fake_imgs.cpu().numpy()
        
        # Denormalize images
        fake_imgs = (fake_imgs + 1) / 2
        fake_imgs = np.clip(fake_imgs, 0, 1)
        
        return fake_imgs
    
# Test function to verify generated images
def test_generation():
    """Test the generation of digits"""
    generator.eval()
    
    # Generate samples for each digit
    for digit in range(10):
        images = generate_digit_images(digit, 5)
        
        # Display the generated images
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            axes[i].imshow(images[i, 0], cmap='gray')
            axes[i].set_title(f'Digit {digit}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'test_digit_{digit}.png')
        plt.close()
    
    print("Test images generated for all digits!")

if __name__ == "__main__":
    print("Starting MNIST GAN training...")
    print(f"Total parameters in Generator: {sum(p.numel() for p in generator.parameters())}")
    print(f"Total parameters in Discriminator: {sum(p.numel() for p in discriminator.parameters())}")
    
    # Train the GAN
    train_gan()
    
    # Test generation
    test_generation()
    
    print("Training and testing completed!")