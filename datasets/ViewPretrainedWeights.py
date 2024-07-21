import numpy as np

def print_npz_structure(file_path):
    # 加载 .npz 文件
    data = np.load(file_path)

    # 打印文件中的键
    print("Keys in the NPZ file:")
    for key in data.keys():
        print("- {}".format(key))

    # 打印每个键对应的数组形状
    print("\nArray shapes:")
    for key in data.keys():
        print("- {}: {}".format(key, data[key].shape))

    # 打印每个键对应的数组内容
    # print("\nArray contents:")
    # for key in data.keys():
    #     print("- {}: \n{}".format(key, data[key]))

# 示例使用
npz_file_path = r"E:\Desktop\TransUNet-main\TransUNet-main\model\R50+ViT-B_16.npz"  # 替换为你的 .npz 文件路径
print_npz_structure(npz_file_path)