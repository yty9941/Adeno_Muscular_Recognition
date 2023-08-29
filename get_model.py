import torch
import torchvision
#获取网络
import config
from model.CMT import CMT
from model.ECA_Net import eca_resnet18, eca_resnet34
from model.ResNet import ResNet
from model.ResT import ResT
from model.SwinTransformer import SwinTransformer


def get_model(model_name):
    if (model_name == 'resnet18'):
        model = ResNet(config.isCbam)
    elif (model_name == 'resnet34'):
        model = ResNet(config.isCbam)
    elif (model_name == 'resnet50'):
        model = ResNet(config.isCbam)
    elif (model_name == 'resnet101'):
        model = ResNet(config.isCbam)
    elif (model_name == 'resnet152'):
        model = ResNet(config.isCbam)
    elif (model_name == 'SwinTransformer'):
        model = SwinTransformer()
    elif (model_name == 'EcaNet18'):
        model = eca_resnet18(num_classes = config.Num_Classes)
    elif (model_name == 'EcaNet34'):
        model = eca_resnet34(num_classes = config.Num_Classes)
    elif (model_name == 'ResT'):
        model = ResT()
    elif (model_name == 'CMT'):
        model = CMT(num_classes = config.Num_Classes)
    else:
        model = None
        print("Model name is wrong!")
    return model