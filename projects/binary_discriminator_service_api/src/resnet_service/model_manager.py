import torch
import os
from torchvision import datasets, models, transforms
import torch.nn as nn
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ModelResNet:

    def __init__(self):
        self.localAddress = os.environ.get("LOCAL_ADDRESS_AI_MODELS", "")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet18(pretrained=False)  # pretrained=False is crucial here
        nr_filters = self.model.fc.in_features
        self.model.fc = nn.Linear(nr_filters, 1)
        self.model.load_state_dict(torch.load(self.localAddress, map_location=torch.device(self.device)))
        self.model.eval()

    def predict(self, sample):
        with torch.no_grad():
            out = torch.sigmoid(self.model(sample))

        if out <= 0.4:
            return "Legible", out
        else:
            return "Non Legible", out
