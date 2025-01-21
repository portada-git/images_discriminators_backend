import os

# Import main FastAPI modules
from fastapi import FastAPI

from api.binary_discriminator import binary_discriminator


app = FastAPI(
    title='Deep Learning Models for Image Classification',
    description='Models for binary image classification (ie: Legible or not legible) '
                'and Multiclass Image Classification',
    version='0.0.1')

app.include_router(
    binary_discriminator.router, tags=['Legibility Discriminator'], prefix='/binary',
    dependencies=[]
)


