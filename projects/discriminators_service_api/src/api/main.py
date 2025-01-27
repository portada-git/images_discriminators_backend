import os

# Import main FastAPI modules
from fastapi import FastAPI

from api.binary_discriminator import binary_discriminator
from api.multiclass_discriminator import multiclass_discriminator


app = FastAPI(
    title='Deep Learning Models for Image Classification',
    description='Models for binary image classification (ie: Legible or not legible) '
                'and Multiclass Image Classification.</br></br><h3>Main endpoints group:</h3></br> '
                '<strong>Legibility Discriminator.</strong> Address a binary '
                'classification problem with two lables: '
                'Legible and Non Legible.</br>'
                '<strong>Image Corrector Discriminator.</strong> Address a multiclass '
                'classification problem with five lables: CURVATURA, INCLINACION, ORIGINAL, RUIDO, and TODO',
    version='0.0.1')

app.include_router(
    binary_discriminator.router, tags=['Legibility Discriminator'], prefix='/binary',
    dependencies=[]
)

app.include_router(
    multiclass_discriminator.router, tags=['Image Corrector Discriminator'], prefix='/multiclass',
    dependencies=[]
)


