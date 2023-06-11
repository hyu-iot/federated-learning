import os
import torch
import torchvision.models as models

import importlib.util
import sys


def load_model(modelname, args, pretrained_path=""):
    torchvision_models = (
        'densenet121', 'mobilenet_v2', 'googlenet', 'resnet18', 'resnet34',
        'resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19'
    )
    
    if modelname in torchvision_models:
        model = getattr(models, modelname)(**args)
        print("Training model: " + modelname)
    elif modelname == 'custom_model':
        spec = importlib.util.spec_from_file_location("custom_model", args['path'])
        foo = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = foo
        spec.loader.exec_module(foo)
        model = foo.CustomModel()
    else:                   # Custom model
        print("Unavailable model name.")
        exit(1)

    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path)['net'])

    return model

