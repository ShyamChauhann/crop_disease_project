import torch
import torch.nn as nn


class SensorAutoencoder(nn.Module):

    def __init__(self, input_dim=8):

        super().__init__()

        # ==========================================
        # ENCODER
        # ==========================================

        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 64),

            nn.ReLU(),

            nn.Linear(64, 128),

            nn.ReLU()
        )


        # ==========================================
        # DECODER
        # ==========================================

        self.decoder = nn.Sequential(

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, input_dim)
        )


    def forward(self, x):

        embedding = self.encoder(x)

        reconstruction = self.decoder(
            embedding
        )

        return embedding, reconstruction