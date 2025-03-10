import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob

# Define the Video Frame Dataset
class VideoFrameDataset(Dataset):
    def __init__(self, video_paths, transform=None):
        self.video_paths = video_paths
        self.transform = transform
        self.frames = []
        
        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                if self.transform:
                    frame = self.transform(frame)
                self.frames.append(frame)
            cap.release()
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        return self.frames[idx]

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Define the Sampling Layer
class Sampling(nn.Module):
    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
    
    def forward(self, z):
        x = F.relu(self.fc(z)).view(-1, 64, 8, 8)
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

# Define the VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(latent_dim)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sampling(mu, logvar)
        return self.decoder(z), mu, logvar, z

# Function to Extract Latent Representations and Save to CSV
def encode_video_to_csv(vae, video_path, output_csv, transform, device):
    cap = cv2.VideoCapture(video_path)
    latents = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            _, mu, _, z = vae(frame)
        latents.append(z.cpu().numpy().flatten())
    cap.release()
    
    df = pd.DataFrame(latents)
    df.to_csv(output_csv, index=False)
    print(f"Latent representations saved to {output_csv}")

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 16
vae = VAE(latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Load Dataset
video_files = glob("./misc/*.mp4")  # Adjust path accordingly
transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32, 32)), transforms.ToTensor()])
dataset = VideoFrameDataset(video_files, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train VAE
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar, _ = vae(batch)
        recon_loss = F.mse_loss(recon, batch)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
torch.save(vae.state_dict(), "vae.pth")

# Encode and Save Latent Representations
encode_video_to_csv(vae, "examples/videos/gt/video1.mp4", "latent_vectors.csv", transform, device)
