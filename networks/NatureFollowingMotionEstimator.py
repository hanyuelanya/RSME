import torch
import numpy as np
import torch.nn as nn
import sys
import time
sys.path.append('./')
from .SkinMotionTotalEncoder import SkinMotionVisionTransformer as SMVT
from .SpatialComponentTotalEncoder import SpatialComponentVisionTransformer as SCVT
from .BidirectionalCrossAttention import BidirectionalCrossAttention as BiCrossA

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.fc3 = nn.Linear(hidden_size2, num_classes)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.fc3(out)
#         return out

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(3*config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size,2)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        nn.init.normal_(self.fc3.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class NatureFollowingMotionEstimator(nn.Module):
    def __init__(self, config):
        super(NatureFollowingMotionEstimator, self).__init__()
        self.SME = SMVT(config,img_size=224, zero_head=False, vis=False)
        self.SME.load_from(weights=np.load(config.pretrained_path))
        self.SCE = SCVT(config)
        self.SCE.load_pretrain()
        self.BCA = BiCrossA(dim=768,
                            heads=8,
                            dim_head=64,
                            context_dim=768)
        self.FCN = Mlp(config)

    def forward(self, ERI, SC1, SC2):

        EncodedSC1 = self.SCE(SC1)  # 1x256x768
        EncodedSC2 = self.SCE(SC2)  # 1x256x768
        EncodedERIStartTime = time.time()
        EncodedERI = self.SME(ERI)  # 1x196x768  ->batchsizex1x768
        EncodedERIEndTime = time.time()
        print("EncodeERI Time:", EncodedERIEndTime-EncodedERIStartTime)
        # 在第0维度上拼接两个张量
        EncodedSC = torch.cat((EncodedSC1, EncodedSC2), dim=1) #512*768 -> 2*768
        # 在结果张量的第0维度上增加一个维度
        # EncodedSC = EncodedSC.unsqueeze(0) # (2*batchsize,512,768)
        # EncodedERI = EncodedERI.unsqueeze(0) # (batchsize,196,768)

        # print("交叉注意力的输入为：",EncodedSC.shape,EncodedERI.shape)
        # attended output should have the same shape as input
        FCNStartTime = time.time()
        SC_out, ERI_out = self.BCA(EncodedSC, EncodedERI)
        # 展平输入张量

        FCN_input = torch.cat((SC_out, ERI_out), dim=1)  #1x708x768 ->batchsizex3x768
        FCN_input = FCN_input.view(FCN_input.shape[0], 1, -1) #batchsize,1,3x768
        # print(FCN_input.shape)
        res = self.FCN(FCN_input)
        FCNEndTime = time.time()
        print("FCN Time:", FCNEndTime-FCNStartTime)
        return res
