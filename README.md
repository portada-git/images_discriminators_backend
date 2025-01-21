[[_TOC_]]

# Introduction
Image Binary Legibility Discriminator.

Given a set of historical press images from different newspapers and previously labelled. Predict the class to which it belongs by minimizing the prediction error.

Task 1: Classify images into readable or not readable.

## ResNet18-based model

Custom Deep ResNet18 with fully connected layer and binary softmax classifier:

![model_design.png](resources%2Fmodel_design.png)

Using Adam as optimizer with 20 epochs and early-stoping

### Evaluation results
| Labels     | Precision | Recal | F1     |
|------------|-----------|------|--------|
| Legible    | 1.0 | 0.8333  | 0.9091 |
| No legible | 0.8 | 1.0 | 0.8889 |

Final weighted  F1-Score is **0.9010**.

### Run on Docker 
```bash
docker-compose up --build binary_discriminator_service_api
```

### Run on Docker with prebuild local image
```bash
docker-compose up --build binary_discriminator_service_api_local
```

### Swagger and Redoc
- Swagger: [localhost:8001/docs](http://localhost:8001/docs)
- Redoc: [localhost:8001/redoc](http://localhost:8001/redoc)