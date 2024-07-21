import math
import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import re
from PIL import Image
import torch.nn.functional as F


def scale_array(arr, scale_factor):
    # 缩放到[-1,1]
    scaled_arr = (arr - np.min(arr)) / scale_factor * 2 - 1
    return scaled_arr


def unscale_array(scaled_arr, scale_factor):
    unscaled_arr = (scaled_arr + 1) / 2 * scale_factor + np.min(scaled_arr)
    return unscaled_arr


# 由于TC被归一化(缩小scale_factor/2倍)，应相应地放大SC
def modify_SC(SC, scale_factor):
    return SC * scale_factor / 2


def scale_displacement_field(displacement_field, target_size=(32, 64, 64)):
    # 创建目标尺寸

    # 构建网格坐标
    grid_y, grid_x, grid_z = torch.meshgrid(torch.linspace(-1, 1, target_size[0]),
                                            torch.linspace(-1, 1, target_size[1]),
                                            torch.linspace(-1, 1, target_size[2]))
    grid = torch.stack((grid_x, grid_y, grid_z), dim=-1).unsqueeze(0)

    # 将网格坐标转换为[-1, 1]的范围
    grid = grid * 2 - 1

    # 将输入位移场和网格坐标调整形状

    displacement_field = displacement_field.unsqueeze(0).permute(0, 4, 1, 2, 3).contiguous()
    grid = grid.contiguous()

    # 使用grid_sample进行缩放
    scaled_displacement_field = F.grid_sample(displacement_field.float(), grid)

    # 调整输出形状 # 最新：不需要手动添加batchsize的维度了
    scaled_displacement_field = scaled_displacement_field.squeeze(0)
    return scaled_displacement_field


def generateERIImage(ee_path, ei_path, realtime_path):
    # 读取三张单通道16位PNG图像
    ee = Image.open(ee_path)
    ei = Image.open(ei_path)
    realtime = Image.open(realtime_path)

    # 将图像转换为numpy数组
    array_ee = np.array(ee)
    array_ei = np.array(ei)
    array_realtime = np.array(realtime)

    # 创建新的RGB图像数组
    new_image_array = np.dstack((array_ee, array_realtime, array_ei))
    # # 调整数据类型
    # new_image_array = (new_image_array).astype(np.uint16)
    # # 创建新的RGB图像
    # new_image = Image.fromarray(new_image_array)
    return new_image_array


def random_rot_90(ERI):
    # 随机选择旋转角度
    k = np.random.randint(0, 3)
    # 对图像和标签进行旋转
    ERI = np.rot90(ERI, k)
    # label = np.rot90(label, k)
    # # 随机选择水平或垂直翻转
    # axis = np.random.randint(0, 2)
    # image = np.flip(image, axis=axis).copy()
    # label = np.flip(label, axis=axis).copy()
    return ERI


# def random_rotate(ERI):
#     # 随机选择旋转角度
#     angle = np.random.randint(-20, 20)
#     # 对图像和标签进行旋转
#     ERI = ndimage.rotate(ERI, angle, order=0, reshape=False)
#     # label = ndimage.rotate(label, angle, order=0, reshape=False)
#     return ERI

# 上面的函数似乎不能处理三通道
def random_rotate(ERI):
    # Randomly select the rotation angle
    angle = np.random.randint(-20, 20)

    # Split the three-channel RGB image into separate channels
    channel_0 = ERI[:, :, 0]
    channel_1 = ERI[:, :, 1]
    channel_2 = ERI[:, :, 2]

    # Rotate each channel separately
    rotated_channel_0 = ndimage.rotate(channel_0, angle, order=0, reshape=False)
    rotated_channel_1 = ndimage.rotate(channel_1, angle, order=0, reshape=False)
    rotated_channel_2 = ndimage.rotate(channel_2, angle, order=0, reshape=False)

    # Stack the rotated channels back into a three-channel RGB image
    rotated_ERI = np.dstack((rotated_channel_0, rotated_channel_1, rotated_channel_2))

    return rotated_ERI


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        # 获取样本中的图像和标签
        ERI, SC1, SC2, TC1, TC2 = sample['ERI'], sample['SC1'], sample['SC2'], sample['TC1'], sample['TC2']
        # print(ERI.shape)
        # if random.random() > 0.5:
        #     # 随机进行旋转和翻转操作
        #     ERI = random_rot_90(ERI)
        # elif random.random() > 0.5:
        #     # 随机进行旋转操作
        #     ERI = random_rotate(ERI)

        if random.random() > 0.5:
            # 随机进行旋转操作
            ERI = random_rotate(ERI)
        x, y = ERI.shape[0],ERI.shape[1]
        # 如果图像尺寸不符合要求，则进行缩放
        if x != self.output_size[0] or y != self.output_size[1]:
            ERI = zoom(ERI, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            # label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        SC1 = scale_displacement_field(torch.from_numpy(SC1).permute(2, 0, 1, 3),
                                       target_size=(32, 64, 64))  # 将256，256，94，3转为1,3,32,64,64
        SC2 = scale_displacement_field(torch.from_numpy(SC2).permute(2, 0, 1, 3),
                                       target_size=(32, 64, 64))  # 将256，256，94，3转为1,3,32,64,64

        # 转换为张量并添加维度(添加维度以进行批处理，这样做的目的是将原始的二维数组转换为具有批次维度的三维张量）
        # 其中批次大小为1。这在处理单个样本时很常见，以便与其他批次处理的张量保持一致
        # 最新：不需要手动添加batchsize的维度了
        # ERI = torch.from_numpy(ERI).permute(2, 0, 1).unsqueeze(0)
        ERI = torch.from_numpy(ERI).permute(2, 0, 1).float()

        label = torch.tensor([[TC1, TC2]])

        sample = {'ERI': ERI, 'SC1': SC1, 'SC2': SC2, 'label': label}
        return sample


crop_info = {
            "case6":{"start":[82,50,None],"end":[464,361,None]},
            "case7":{"start":[82,50,None],"end":[464,361,None]},
            "case8":{"start":[77,27,None],"end":[421,329,None]},
            "case9":{"start":[80,65,None],"end":[422,356,None]},
            "case10":{"start":[110,65,None],"end":[420,375,None]}
            }

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        # listdir应为./lists  data_dir应为D:\k-SpaceCTReconstruction
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            # 获取皮肤深度图文件名
            # 使用正则表达式匹配case_num和phase_num
            match = re.match(r"case(\d+)-(\d+)", self.sample_list[idx])
            case_num = ""
            phase_num = ""
            if match:
                case_num = match.group(1)
                phase_num = match.group(2)
            else:
                assert "Invalid train list string format."

            # 整理数据路径
            skin_ee_path = os.path.join(self.data_dir, "Skins/DepthImage/4DCT_Case" + case_num, "50.png")
            skin_ei_path = os.path.join(self.data_dir, "Skins/DepthImage/4DCT_Case" + case_num, "00.png")
            skin_realime_path = os.path.join(self.data_dir, "Skins/DepthImage/4DCT_Case" + case_num,
                                             phase_num + ".png")
            components_path = os.path.join(self.data_dir, "DVFs/4DCT/Case" + case_num, "principal_components.npz")
            scale_factors_path = os.path.join(self.data_dir, "DVFs/4DCT/Case" + case_num, "TCs_scale_factor.npz")

            # 加载数据
            ERI = generateERIImage(skin_ee_path, skin_ei_path, skin_realime_path)  # 获得ERI(224x224x3)
            components = np.load(components_path)
            scale_factors = np.load(scale_factors_path)['TCs_scale_factor']
            SC1 = components["principal_components"][0]  # 将所有样本主成分归一化到正方向，相对的在下面的代码也对TC进行了方向处理
            SC2 = components["principal_components"][1]

            #去除case6-case10的主成分冗余部分
            if int(case_num)>5:
              start = crop_info["case"+case_num]["start"]
              end = crop_info["case"+case_num]["end"]
              SC1 = SC1[start[0]:end[0],start[1]:end[1],:,:]
              SC2 = SC2[start[0]:end[0],start[1]:end[1],:,:]

            # print(scale_factors)
            scale_factor1 = scale_factors[0]  # 负号代表需要翻转，应先取绝对值作为缩放因子将TCs缩放至【-1，1】之后再翻转
            scale_factor2 = scale_factors[1]

            SC1 *= (1 if scale_factor1 > 0 else -1)
            SC2 *= (1 if scale_factor1 > 0 else -1)
            SC1 = modify_SC(SC1,abs(scale_factor1))
            SC2 = modify_SC(SC2, abs(scale_factor2))
            TCs = components["coefficients"]
            TCs[:, 0] = scale_array(TCs[:, 0], abs(scale_factor1))
            TCs[:, 1] = scale_array(TCs[:, 1], abs(scale_factor2))

            # 对于50时相作为实时，位移场的各分量应该为0（coefficients中没有存储），但这一点应继续考虑，因为50时相的分量不一定全为0
            if phase_num == '50':
                TC1, TC2 = 0, 0
            else:
                # coefficients中存的依次是50至00 10 20 30 40 60 70 80 90的位移场的系数
                index = int(phase_num[0]) if int(phase_num) < 50 else int(phase_num[0]) - 1
                TC1 = TCs[index][0] * (1 if scale_factor1 > 0 else -1)
                TC2 = TCs[index][1] * (1 if scale_factor2 > 0 else -1)
            # image, label = data['image'], data['label']
        else:
            # 获取皮肤深度图文件名
            # 使用正则表达式匹配case_num和phase_num
            match = re.match(r"case(\d+)-(\d+)", self.sample_list[idx])
            case_num = ""
            phase_num = ""
            if match:
                case_num = match.group(1)
                phase_num = match.group(2)
            else:
                assert "Invalid train list string format."

            # 整理数据路径
            skin_ee_path = os.path.join(self.data_dir, "Skins/DepthImage/4DCT_Case" + case_num, "50.png")
            skin_ei_path = os.path.join(self.data_dir, "Skins/DepthImage/4DCT_Case" + case_num, "00.png")
            skin_realime_path = os.path.join(self.data_dir, "Skins/DepthImage/4DCT_Case" + case_num,
                                             phase_num + ".png")
            components_path = os.path.join(self.data_dir, "DVFs/4DCT/Case" + case_num, "principal_components.npz")
            scale_factors_path = os.path.join(self.data_dir, "DVFs/4DCT/Case" + case_num, "TCs_scale_factor.npz")
            # 加载数据
            ERI = generateERIImage(skin_ee_path, skin_ei_path, skin_realime_path)  # 获得ERI(224x224x3)
            components = np.load(components_path)
            scale_factors = np.load(scale_factors_path)['TCs_scale_factor']
            SC1 = components["principal_components"][0]
            SC2 = components["principal_components"][1]

            #去除case6-case10的主成分冗余部分
            if int(case_num)>5:
              start = crop_info["case"+case_num]["start"]
              end = crop_info["case"+case_num]["end"]
              SC1 = SC1[start[0]:end[0],start[1]:end[1],:,:]
              SC2 = SC2[start[0]:end[0],start[1]:end[1],:,:]

            scale_factor1 = scale_factors[0]  # 负号代表需要翻转，应先取绝对值作为缩放因子将TCs缩放至【-1，1】之后再翻转
            scale_factor2 = scale_factors[1]
            SC1 *= (1 if scale_factor1 > 0 else -1)
            SC2 *= (1 if scale_factor1 > 0 else -1)
            SC1 = modify_SC(SC1,abs(scale_factor1))
            SC2 = modify_SC(SC2, abs(scale_factor2))
            TCs = components["coefficients"]
            TCs[:, 0] = scale_array(TCs[:, 0], abs(scale_factor1))
            TCs[:, 1] = scale_array(TCs[:, 1], abs(scale_factor2))
            # 对于50时相作为实时，位移场的各分量应该为0（coefficients中没有存储），但这一点应继续考虑，因为50时相的分量不一定全为0
            if phase_num == '50':
                TC1, TC2 = 0, 0
            else:
                # coefficients中存的依次是50至00 10 20 30 40 60 70 80 90的位移场的系数
                index = int(phase_num[0]) if int(phase_num) < 50 else int(phase_num[0]) - 1
                TC1 = TCs[index][0] * (1 if scale_factor1 > 0 else -1)
                TC2 = TCs[index][1] * (1 if scale_factor1 > 0 else -1)

        sample = {'ERI': ERI, 'SC1': SC1, 'SC2': SC2, 'TC1': TC1, 'TC2': TC2}
        if self.transform:
            # 应用数据增强
            # 这里的transforme就是RandomGenerate，可以进行数据增强和标准化
            sample = self.transform(sample)
        # print("shape为", sample['SC1'].shape)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
