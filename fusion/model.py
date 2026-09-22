import torch
import torch.nn as nn
from torchvision import models

class MultimodalModel(nn.Module):

    def __init__(
        self,
        num_classes=65
    ):

        super().__init__()


        # ==========================================
        # IMAGE ENCODER
        # ==========================================

        resnet = models.resnet50(
            weights="DEFAULT"
        )


        # Remove original classifier
        self.image_encoder = nn.Sequential(
            *list(resnet.children())[:-1]
        )


        # ==========================================
        # SENSOR ENCODER
        # ==========================================

        self.sensor_encoder = nn.Sequential(

            nn.Linear(8, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 128),

            nn.ReLU()
        )


        # ==========================================
        # FUSION CLASSIFIER
        # ==========================================

        self.classifier = nn.Sequential(

            nn.Linear(
                2048 + 128,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                num_classes
            )
        )


    def forward(
        self,
        image,
        sensor
    ):


        # ==========================================
        # IMAGE FEATURES
        # ==========================================

        image_features = self.image_encoder(
            image
        )


        image_features = (
            image_features.flatten(1)
        )


        # Shape:
        # [batch, 2048]


        # ==========================================
        # SENSOR FEATURES
        # ==========================================

        sensor_features = (
            self.sensor_encoder(sensor)
        )


        # Shape:
        # [batch, 128]


        # ==========================================
        # LATE FUSION
        # ==========================================

        fused_features = torch.cat(
            [
                image_features,
                sensor_features
            ],
            dim=1
        )


        # Shape:
        # [batch, 2176]


        # ==========================================
        # CLASSIFICATION
        # ==========================================

        output = self.classifier(
            fused_features
        )


        return output