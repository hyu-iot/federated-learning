import os
import torch
import torchvision.models as models


def load_model(modelname, args, pretrained_path=""):
    torchvision_models = (
        'densenet121', 'mobilenet_v2', 'googlenet', 'resnet18', 'resnet34',
        'resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
    )
    
    if modelname in torchvision_models:
        model = getattr(models, modelname)(**args)
    else:                   # Custom model
        model = None

    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path)['net'])

    return model

