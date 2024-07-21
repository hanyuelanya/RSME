import numpy as np

for case in range(1,11):
    file_path = r"D:\k-SpaceCTReconstruction\DVFs\4DCT\Case"+str(case)+"\principal_components.npz"
    data = np.load(file_path)

    # 查看文件中的所有变量名
    variable_names = list(data.keys())
    print("Variable names:", variable_names)

    # 输出所有变量的值
    for variable_name in variable_names:
        variable_value = data[variable_name]
        # print(f"{variable_name}:")
        print(np.mean(variable_value))
        print()