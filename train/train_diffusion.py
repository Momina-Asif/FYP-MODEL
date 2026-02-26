import torch
from models.diffusion_unet import UNet

model = UNet(in_channels=6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

image = torch.randn(1, 3, 256, 256)
mask = torch.randn(1, 3, 256, 256)
noise = torch.randn_like(image)

noisy_img = image + noise
input_tensor = torch.cat([noisy_img, mask], dim=1)

pred_noise = model(input_tensor)
loss = torch.nn.functional.mse_loss(pred_noise, noise)

loss.backward()
optimizer.step()
