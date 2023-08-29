from torch import nn
from torchvision import models
import config

from model.cbam_module import CBAM

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class ResNet(nn.Module):
    def __init__(self, isCbam = False):
        super(ResNet, self).__init__()
        self.isCbam = isCbam
        self.base_model_name = config.Model_Backbone
        self.model_dict = {"resnet18": models.resnet18(pretrained = config.isPretrained),
                           "resnet34": models.resnet34(pretrained = config.isPretrained),
                           "resnet50": models.resnet50(pretrained = config.isPretrained),
                           "resnet101": models.resnet101(pretrained = config.isPretrained)}
        base_model = self.getBaseModel()
        base_model.conv1 = nn.Conv2d(config.In_Channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)  # 修改输入通道数
        num_features = base_model.fc.in_features  # 输出的通道数
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        if(self.isCbam):
            self.cbam = CBAM(num_features)
        # projection MLP for ResNet Model
        # self.attention = nn.MultiheadAttention(num_features, num_heads = 1, batch_first = True)  # 可以调整num_heads来增加注意头数
        self.mlp = nn.Linear(num_features, 10, bias = True)
        self.dropout = nn.Dropout(p = 0.2)
        self.relu = nn.ReLU(inplace = True)
        self.project_head = nn.Linear(10, config.Num_Classes, bias = False)
        self.sigmoid = nn.Sigmoid()
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std = .02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def getBaseModel(self):
        try:
            base_model = self.model_dict[self.base_model_name]
            print("Image feature extractor:", self.base_model_name)
        except:
            raise ("Invalid model name. Check the config file!")
        return base_model

    def forward(self, image):
        out = self.backbone(image)
        # print("hidden", hidden.size())
        # print("hidden:",hidden.size())
        # hidden = hidden.squeeze(3).permute(0,2,1)
        # out, _ = self.attention(hidden, hidden, hidden)
        # out = out.squeeze()
        # hidden = hidden.squeeze()
        if(self.isCbam):
            out = self.cbam(out)
        out = out.squeeze()

        out = self.mlp(out)
        out = self.relu(out)
        out = self.dropout(out)
        # print("out1:", out.size())
        # print("out2:", out.size())
        out = self.project_head(out)
        out = self.sigmoid(out)
        # print("out3:", out.size())
        return out