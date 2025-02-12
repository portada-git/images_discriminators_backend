import torch
import os
from torchvision import datasets, models, transforms
import torch.nn as nn
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

class ModelWithSigmoid(nn.Module):
    def __init__(self, model, labels):
        super(ModelWithSigmoid, self).__init__()
        self.model = model
        self.sigmoid = nn.Sigmoid()  # Apply softmax along the class dimension

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x


class ModelMulticlassResNet:

    def __init__(self):
        self.localAddress = f"""{os.environ.get("LOCAL_ADDRESS_AI_MODELS", 
                                                "")}/model_weights_model_sigmoid_todo_new.pth"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_rest_net_pre = models.resnet18(pretrained=True)
        nr_filters = model_rest_net_pre.fc.in_features
        model_rest_net_pre.fc = nn.Linear(nr_filters, 4)
        self.model = ModelWithSigmoid(model_rest_net_pre, 4)  #
        self.model.load_state_dict(torch.load(self.localAddress, map_location=torch.device(self.device)))
        self.model.eval()

    def predict(self, sample):
        thresholds = [0.45, 0.25, 0.35, 0.55] # 0.2, 0.3, 0.15, 0.25 (old version)

        with torch.no_grad():
            yhat = self.model(sample)
            predictions = yhat.cpu().numpy()

            for i in range(predictions.shape[0]):
                predicted_classes = []
                predicted_classes_ponderations = []
                for j in range(predictions.shape[1]):
                    if predictions[i][j] > thresholds[j]:
                        predicted_classes.append(j)
                        predicted_classes_ponderations.append(predictions[i][j])
                if len(predicted_classes) == 0:
                    predicted_class = 2
                elif len(predicted_classes) == 1:
                    predicted_class = predicted_classes[0]
                elif len(predicted_classes) > 1 and 2 not in predicted_classes:
                    predicted_classes_ponderations = np.array(predicted_classes_ponderations)
                    score = np.sum(predicted_classes_ponderations)
                    if score > 1.35:  # 1.35
                        predicted_class = 4
                    else:
                        max_index = np.argmax(predictions[i])
                        predicted_class = max_index
                else:
                    max_index = np.argmax(predictions[i])
                    predicted_class = max_index

        class_mapping = {
            0: "CURVATURA",
            1: "INCLINACION",
            2: "ORIGINAL",
            3: "RUIDO",
            4: "TODO"
        }

        suggested_transformation_indexes = []
        if predicted_class == 4:
            suggested_transformation = [class_mapping[i] for i in predicted_classes]
            suggested_transformation_indexes = predicted_classes
        else:
            suggested_transformation = [class_mapping[predicted_class]]
            suggested_transformation_indexes.append(predicted_class)

        return class_mapping[predicted_class], predicted_class, yhat.cpu().numpy()[0].tolist(), \
            suggested_transformation, suggested_transformation_indexes
