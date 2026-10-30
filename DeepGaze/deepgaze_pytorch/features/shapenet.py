"""
This code was adapted from: https://github.com/rgeirhos/texture-vs-shape
"""
import os
import sys
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
import torchvision.models
from torch.utils import model_zoo

from .normalizer import Normalizer


import torchvision.models as models


def load_model(model_name):
  # Bypass dead Bitbucket links by using reliable standard torchvision weights
  if "resnet50" in model_name:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(OrderedDict([("module", model)]))
    return model

  elif "vgg16" in model_name:
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.features = torch.nn.DataParallel(model.features)
    return model

  elif "alexnet" in model_name:
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    return model

  else:
    raise ValueError(f"Unsupported model name: {model_name}")


class RGBShapeNetA(nn.Sequential):
    def __init__(self):
        super(RGBShapeNetA, self).__init__()
        self.shapenet = load_model("resnet50_trained_on_SIN")
        self.normalizer = Normalizer()
        super(RGBShapeNetA, self).__init__(self.normalizer, self.shapenet)



class RGBShapeNetB(nn.Sequential):
    def __init__(self):
        super(RGBShapeNetB, self).__init__()
        self.shapenet = load_model("resnet50_trained_on_SIN_and_IN")
        self.normalizer = Normalizer()
        super(RGBShapeNetB, self).__init__(self.normalizer, self.shapenet)


class RGBShapeNetC(nn.Sequential):
    def __init__(self):
        super(RGBShapeNetC, self).__init__()
        self.shapenet = load_model("resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN")
        self.normalizer = Normalizer()
        super(RGBShapeNetC, self).__init__(self.normalizer, self.shapenet)



