# Import modules from FastAPI
from fastapi import APIRouter, File, UploadFile
from typing_extensions import Annotated
from pydantic import BaseModel
from PIL import Image

# Set the API Router
router = APIRouter()
from resnet_service.model_manager import ModelResNet
import torchvision
from torchvision import transforms
import torch

model = ModelResNet()


class PredictionSchemaResponse(BaseModel):
    label: str
    score: float


@router.post('/predict')
async def predict(file: Annotated[UploadFile,
                            File(description="A valid image")], threshold: float = 0.4):
    """
   Predicts if an image is legible or not using a trained resnet model.

   Parameters:
       file (UploadFile): A valid image file uploaded by the user.
       threshold: Float value that represents score's dirty to overcome for Non-Legible.

   Returns:
       PredictionSchemaResponse: A JSON object containing the predicted text and its score of dirty.
       If dirty overcome threshold Non-legible is considered.
   """
    test_transforms = transforms.Compose([transforms.Resize((1024, 1024)),
                                          transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

    image = Image.open(file.file).convert('RGB')
    img_tensor = test_transforms(image)
    img_tensor = torch.unsqueeze(img_tensor, dim=0).to(model.device)
    label, score = model.predict(img_tensor, threshold)

    return PredictionSchemaResponse(label=label, score=score)
