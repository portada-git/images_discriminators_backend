import torch
import os
from torchvision import datasets, models, transforms
import torch.nn as nn
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ModelResNet:

    def __init__(self):
        self.localAddressPages = f"""{os.environ.get("LOCAL_ADDRESS_AI_MODELS", "")}/binary_discriminator_pages.pth"""
        self.localAddressBlocks = f"""{os.environ.get("LOCAL_ADDRESS_AI_MODELS", "")}/binary_discriminator_blocks.pth"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet18(pretrained=False)  # pretrained=False is crucial here
        nr_filters = self.model.fc.in_features
        self.model.fc = nn.Linear(nr_filters, 1)
        self.model.load_state_dict(torch.load(self.localAddressPages, map_location=torch.device(self.device)))
        self.model.eval()

        self.model_blocks = models.resnet18(pretrained=False)  # pretrained=False is crucial here
        nr_filters = self.model_blocks.fc.in_features
        self.model_blocks.fc = nn.Linear(nr_filters, 1)
        self.model_blocks.load_state_dict(torch.load(self.localAddressBlocks, map_location=torch.device(self.device)))
        self.model_blocks.eval()

    def predict_pages(self, sample, threshold):
        with torch.no_grad():
            out = torch.sigmoid(self.model(sample))

        if out <= threshold:
            return "Legible", out
        else:
            return "Non Legible", out

    def predict_blocks(self, sample, threshold):
        with torch.no_grad():
            out = torch.sigmoid(self.model_blocks(sample))

        if out <= threshold:
            return "Legible", out
        else:
            return "Non Legible", out
