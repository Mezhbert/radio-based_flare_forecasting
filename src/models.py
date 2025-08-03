import torch
import torch.nn as nn


class ConvAE(nn.Module):
    """
    Cверточный автоэнкодер для реконструкции двухканальных сканов.
    """
    def __init__(self, input_channels: int, latent_dim: int, img_height: int, img_width: int):
        super(ConvAE, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, img_height, img_width)
            self._pre_latent_shape = self.encoder(dummy_input).shape
        self.flattened_size = self._pre_latent_shape[1] * self._pre_latent_shape[2] * self._pre_latent_shape[3]
        
        self.fc_encoder = nn.Linear(self.flattened_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flattened_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, input_channels, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(x)
        batch_size, channels, h, w = enc_out.shape
        latent = self.fc_encoder(enc_out.view(batch_size, -1))
        dec_in = self.fc_decoder(latent).view(batch_size, channels, h, w)
        reconstruction = self.decoder(dec_in)

        if reconstruction.shape[2:] != x.shape[2:]:
            reconstruction = nn.functional.interpolate(
                reconstruction,
                size=(self.img_height, self.img_width),
                mode='bilinear',
                align_corners=False
            )
            
        return reconstruction, latent
