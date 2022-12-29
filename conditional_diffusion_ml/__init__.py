# Import modules
from conditional_diffusion_ml.utils import *
from conditional_diffusion_ml.modules import UNet
import logging
import torch
import torch.nn as nn
from tqdm import tqdm

# Initialize logger
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


class Diffusion:
    def __init__(self, dimensions, features, beta_start=1e-4, beta_end=0.02, device='cuda', noise_steps=1000):

        # Build U-Net model
        self.dimensions = dimensions
        self.features = features
        self.model = UNet(channels=features).to(device)

        # Set diffusion parameters
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # Prepare noise schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def generate(self, samples):

        # Switch model into evaluation mode
        logging.info(f'Generating {samples} samples...')
        self.model.eval()
        with torch.no_grad():

            # Start with random noise as input
            x = torch.randn((samples, 1, *self.dimensions)).to(self.device)

            # Incrementally denoise input
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

                # Set denoising parameters for current time step
                t = (torch.ones(samples) * i).long().to(self.device)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # Partially remove noise from input
                predicted_noise = self.model(x, t)
                predicted_noise_scaled = ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                noise_scaled = torch.sqrt(beta) * noise
                x = 1 / torch.sqrt(alpha) * (x - predicted_noise_scaled) + noise_scaled

        # Rescale input
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        # Switch model back into training mode
        self.model.train()

        return x

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def train(self, dataloader, epochs=500, learning_rate=3e-4):

        # Set optimizer and loss function
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        mse = nn.MSELoss()

        # Train for multiple epochs
        for epoch in range(epochs):
            logging.info(f'Starting epoch {epoch}:')

            # Train for each batch in dataloader
            progress_bar = tqdm(dataloader)
            for i, (images, _) in enumerate(progress_bar):
                images = images.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                predicted_noise = self.model(x_t, t)
                loss = mse(noise, predicted_noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix(MSE=loss.item())

            sampled_images = self.generate(16)
            save_images(sampled_images, f'results/Epoch-{epoch}.jpg')
            # torch.save(model.state_dict(), os.path.join('models', args.run_name, f'ckpt.pt'))


if __name__ == '__main__':
    pass
