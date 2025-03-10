import torch
import numpy as np
import pandas as pd
import cv2
import imageio
from partial_vae import Decoder, VAE
import matplotlib.pyplot as plt

# Load Trained Decoder
def load_vae_decoder(latent_dim, device, model_path="vae.pth"):
    vae = VAE(latent_dim).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    return vae.decoder

# Decode CSV Latent Space to Video
def decode_csv_to_video(latent_csv, output_video, latent_dim=16, frame_size=(32, 32), device="cuda"):
    # Load latent vectors
    df = pd.read_csv(latent_csv)
    latent_vectors = torch.tensor(df.values, dtype=torch.float32).to(device)

    # Load trained decoder
    decoder = load_vae_decoder(latent_dim, device)

    # Decode frames
    frames = []
    with torch.no_grad():
        for z in latent_vectors:
            z = z.unsqueeze(0)  # Add batch dimension
            frame = decoder(z).squeeze(0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8).squeeze(0)  # Convert to grayscale
            frame = cv2.resize(frame, frame_size)  # Resize to original frame size
            plt.imshow(frame, cmap='gray')
            plt.show()
            frames.append(frame)
            

    # Save as video
    imageio.mimsave(output_video, frames, fps=30)
    print(f"Reconstructed video saved to {output_video}")

# Run Decoding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decode_csv_to_video("latent_vectors.csv", "reconstructed.mp4", device=device)
