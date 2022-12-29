import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from conditional_diffusion_ml.utils import *
from conditional_diffusion_ml.modules import UNet
import logging
# from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train(self, dataloader, epochs=500, image_size=32, lr=3e-4):
        model = UNet(channels=1).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        mse = nn.MSELoss()
        # diffusion = Diffusion(img_size=image_size, device=self.device)
        # l = len(dataloader)

        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader)
            for i, (images, _) in enumerate(pbar):
                images = images.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                predicted_noise = model(x_t, t)
                loss = mse(noise, predicted_noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(MSE=loss.item())
                # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

            sampled_images = self.sample(model, n=images.shape[0])
            save_images(sampled_images, f"results/Epoch-{epoch}.jpg")
            # torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


# def launch():
#     import argparse
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.run_name = "DDPM_Uncondtional"
#     args.epochs = 500
#     args.batch_size = 12
#     args.image_size = 64
#     args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
#     args.device = "cuda"
#     args.lr = 3e-4
#     train(args)


if __name__ == '__main__':
    pass
    # launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()