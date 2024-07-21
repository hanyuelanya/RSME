# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from .ResNet3D import resnet50


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SpatialComponentEmbeddings(nn.Module):
    """构建嵌入层，包括图像的patch和位置嵌入。
    """
    def __init__(self, config, img_size, in_channels=3):
        super(SpatialComponentEmbeddings, self).__init__()
        self.hybrid = None  # 初始化hybrid属性
        self.config = config  # 配置参数
        # img_size = _pair(img_size)  # 图像尺寸转换为tuple

        if config.patches.get("grid") is not None:  # 如果使用CNN+Transformer混合模型
            # grid_size = config.patches["grid"]  # 获取网格大小参数
            # patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])  # 计算patch的大小
            # patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)  # 计算实际的patch大小
            # n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  # 计算总的patch数目
            patch_size = 1 #可以调整
            n_patches = 4*8*8
            self.hybrid = True  # 设置hybrid为True
        else: # 如果只使用Transformer模型（消融实验）
            # patch_size = _pair(config.patches["size"])  # 获取patch的大小参数
            # n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 计算总的patch数目
            patch_size = 1
            n_patches = 4 * 8 * 8
            self.hybrid = False  # 设置hybrid为False

        #这里的ResNet就是Transformer前面的卷积层
        if self.hybrid:  # 如果使用CNN+Transformer混合模型
            # self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)  # 初始化CNN
            self.hybrid_model = resnet50()  # 初始化CNN
            #in_channels = self.hybrid_model.width * 16  # 更新输入通道数
            in_channels = 512*4  # 更新输入通道数,对于ResNet50是2048，对于38等是512

        #这里的Conv是用于生成Transformer的patch块，卷积操作相当于线性投影
        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)  # 初始化patch嵌入层

        # TODO
        # 用于分类的首个patch
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # self.mlp_head = nn.Linear(dim, num_classes)
        # TODO
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))  # 初始化位置嵌入E_pos

        self.dropout = Dropout(config.transformer["dropout_rate"])  # 初始化dropout层

    def expand_cls_to_batch(self, batch):
      """
      Args:
          batch: batch size
      Returns: cls token expanded to the batch size
      """
      return self.cls_token.expand([batch, -1, -1])

    def forward(self, x):
        if self.hybrid:  # 如果使用混合模型
            # x, features = self.hybrid_model(x)  # 获得CNN的输出
            x = self.hybrid_model(x)  # 获得CNN的输出
        else:
            features = None  # 不使用混合模型，特征为空
        x = self.patch_embeddings(x)  # 应用patch嵌入层
        x = x.flatten(2)  # 将嵌入的维度展平
        x = x.transpose(-1, -2)  # 转置维度顺序
        batch = x.shape[0]
        # TODO
        x = torch.cat((self.expand_cls_to_batch(batch), x), dim=1)  # 将分类patch置于patches首
        # TODO

        embeddings = x + self.position_embeddings  # 加上位置嵌入
        embeddings = self.dropout(embeddings)  # 应用dropout
        return embeddings  # 返回嵌入结果和特征信息


#Transformer的Block块
class SpatialComponentBlock(nn.Module):

    def __init__(self, config, vis):
        super(SpatialComponentBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    # 对于SCEncoder，没有合适的预训练模型，且任务不同
    # def load_from(self, weights, n_block):
    #     ROOT = f"Transformer/encoderblock_{n_block}"
    #     with torch.no_grad():
    #         query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
    #         key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
    #         value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
    #         out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
    #
    #         query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
    #         key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
    #         value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
    #         out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)
    #
    #         self.attn.query.weight.copy_(query_weight)
    #         self.attn.key.weight.copy_(key_weight)
    #         self.attn.value.weight.copy_(value_weight)
    #         self.attn.out.weight.copy_(out_weight)
    #         self.attn.query.bias.copy_(query_bias)
    #         self.attn.key.bias.copy_(key_bias)
    #         self.attn.value.bias.copy_(value_bias)
    #         self.attn.out.bias.copy_(out_bias)
    #
    #         mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
    #         mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
    #         mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
    #         mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()
    #
    #         self.ffn.fc1.weight.copy_(mlp_weight_0)
    #         self.ffn.fc2.weight.copy_(mlp_weight_1)
    #         self.ffn.fc1.bias.copy_(mlp_bias_0)
    #         self.ffn.fc2.bias.copy_(mlp_bias_1)
    #
    #         self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
    #         self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
    #         self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
    #         self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class SpatialComponentEncoder(nn.Module):
    def __init__(self, config, vis):
        super(SpatialComponentEncoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = SpatialComponentBlock(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class SpatialComponentTransformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(SpatialComponentTransformer, self).__init__()
        self.embeddings = SpatialComponentEmbeddings(config, img_size=img_size)
        self.encoder = SpatialComponentEncoder(config, vis)

    def forward(self, input_ids):
        embedding_output= self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights




class SpatialComponentVisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SpatialComponentVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = SpatialComponentTransformer(config, img_size, vis)
        # self.decoder = DecoderCup(config)
        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)

        #得到的x才是Transformer的输入，features是CNN每层的输出（用于U-Net）
        x, attn_weights = self.transformer(x)  # (B, n_patch, hidden)
        # x = self.decoder(x, features)
        # logits = self.segmentation_head(x)
        #下面还需要空间注意力和全连接
        # todo
        x = x[:, 0, :]  # 只获取分类patch用于后续FC输入
        x = x.unsqueeze(1)
        # todo
        return x

    #改为只加载3DResNet50的预训练权重
    def load_pretrain(self):
        with torch.no_grad():
            net_dict = self.transformer.embeddings.hybrid_model.state_dict()
            print('loading 3D ResNet pretrained model {}'.format(self.config.ResNet3D_pretrain_path))
            pretrain = torch.load(self.config.ResNet3D_pretrain_path)
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
            net_dict.update(pretrain_dict)
            self.transformer.embeddings.hybrid_model.load_state_dict(net_dict)
            # res_weight = weights
            # self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            # self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            #
            # self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            # self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            #
            # posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            #
            # posemb_new = self.transformer.embeddings.position_embeddings
            # if posemb.size() == posemb_new.size():
            #     self.transformer.embeddings.position_embeddings.copy_(posemb)
            # elif posemb.size()[1]-1 == posemb_new.size()[1]:
            #     posemb = posemb[:, 1:]
            #     self.transformer.embeddings.position_embeddings.copy_(posemb)
            # else:
            #     logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
            #     ntok_new = posemb_new.size(1)
            #     if self.classifier == "seg":
            #         _, posemb_grid = posemb[:, :1], posemb[0, 1:]
            #     gs_old = int(np.sqrt(len(posemb_grid)))
            #     gs_new = int(np.sqrt(ntok_new))
            #     print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
            #     posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
            #     zoom = (gs_new / gs_old, gs_new / gs_old, 1)
            #     posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
            #     posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
            #     posemb = posemb_grid
            #     self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
            #
            # # Encoder whole
            # for bname, block in self.transformer.encoder.named_children():
            #     for uname, unit in block.named_children():
            #         unit.load_from(weights, n_block=uname)
            #
            # if self.transformer.embeddings.hybrid:
            #     self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
            #     gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
            #     gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
            #     self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
            #     self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
            #
            #     for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
            #         for uname, unit in block.named_children():
            #             unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


