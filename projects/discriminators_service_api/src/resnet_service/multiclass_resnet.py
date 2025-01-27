import torch
import os
from torchvision import datasets, models, transforms
import torch.nn as nn
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ModelWithSoftmax(nn.Module):
    def __init__(self, model, labels):
        super(ModelWithSoftmax, self).__init__()
        self.model = model
        self.fc = nn.Linear(1000, labels)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along the class dimension

    def forward(self, x):
        x = self.model(x)
        x=self.fc(x)
        x = self.softmax(x)
        return x


class ModelMulticlassResNet:

    def __init__(self):
        self.localAddress = f"""{os.environ.get("LOCAL_ADDRESS_AI_MODELS", "")}/model_weights_multiclass_classification.pth"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_rest_net_pre = models.resnet18(pretrained=False)  # pretrained=False is crucial here
        self.model = ModelWithSoftmax(model_rest_net_pre, 5)  #
        self.model.load_state_dict(torch.load(self.localAddress, map_location=torch.device(self.device)))
        self.model.eval()

    def predict(self, sample):
        with torch.no_grad():
            yhat = self.model(sample)
            predicted_class = torch.argmax(yhat, dim=1).cpu().numpy()[0]

        class_mapping = {
            0: "CURVATURA",
            1: "INCLINACION",
            2: "ORIGINAL",
            3: "RUIDO",
            4: "TODO"
        }

        return class_mapping[predicted_class], yhat.cpu().numpy()[0].tolist()
