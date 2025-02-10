# Import modules from FastAPI
from fastapi import APIRouter, File, UploadFile
from typing_extensions import Annotated
from pydantic import BaseModel
from PIL import Image

# Set the API Router
router = APIRouter()
from resnet_service.multiclass_resnet import ModelMulticlassResNet
import torchvision
from torchvision import transforms
import torch
import time
import os

model = ModelMulticlassResNet()


class PredictionSchemaResponseMulticlass(BaseModel):
    label: str
    discrete_label: int
    score: list[float]
    suggested_transformation: list[str]
    suggested_transformation_indexes: list[int]
    elapsed_time: float


@router.post('/predict')
async def predict(file: Annotated[UploadFile,
                            File(description="A valid image")]):
    """
        Predicts if a suitable image's transformation for increasing OCR accuracy, such:

            0: CURVATURA,
            1: INCLINACION,
            2: ORIGINAL,
            3: RUIDO,
            4: TODO

        Model is a pre-trained ResNet with fully-connected layer and softmax on top.

        Parameters:
            file (UploadFile): A valid image file uploaded by the user.

        Returns:
            PredictionSchemaResponse: A JSON object containing the predicted text and its score of dirty.
            If dirty overcome threshold Non-legible is considered.
    """

    start_time = time.time()
    test_transforms = transforms.Compose([transforms.Resize((1024, 1024)),
                                          transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

    image = Image.open(file.file).convert('RGB')
    img_tensor = test_transforms(image)
    img_tensor = torch.unsqueeze(img_tensor, dim=0).to(model.device)
    label, discrete_label, score, suggested_transformation, suggested_transformation_indexes = model.predict(img_tensor)
    end_time = time.time()

    elapsed_time = end_time - start_time
    # os.remove(os.getcwd()+"/"+file.filename)
    return PredictionSchemaResponseMulticlass(label=label, discrete_label=discrete_label, score=score,
                                              elapsed_time=elapsed_time,
                                              suggested_transformation=suggested_transformation,
                                              suggested_transformation_indexes=suggested_transformation_indexes)
