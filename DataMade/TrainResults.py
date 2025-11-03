# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 20:25:08 2024

@author: XCH
"""

import os
import pandas as pd

# 定义文件路径
base_folder = r"D:\2023-Time\Data+Code\01\UDA-Code\wandb"

# 创建一个空的DataFrame，用于存储所有txt文件的内容
all_data = pd.DataFrame()

# 遍历每个子文件夹
for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)
    # print(subfolder_path)
    if os.path.isdir(subfolder_path):
        # 找到files\media\table文件夹
        table_folder = os.path.join(subfolder_path, "files", "media", "table")
        # print(table_folder)
        if os.path.exists(table_folder):
            # 遍历所有txt文件
            txt_files = sorted(os.listdir(table_folder))
            # print(txt_files)
            for txt_file in txt_files:
                txt_file_path = os.path.join(table_folder, txt_file)
                # print(txt_file_path)
                # 读取txt文件内容
                with open(txt_file_path, "r", encoding="utf-8") as file:
                    content = file.readlines()
                    # print(content)
                # 将内容转换为DataFrame
                df = pd.DataFrame(content)
                # 将内容追加到all_data中
                all_data = pd.concat([all_data, df])

# 将所有数据写入Excel
all_data.to_excel(r"D:\2023-Time\Data+Code\01\UDA-Code\052AdvSKM-CtoK.xlsx", index=False, header=False)


file_path = r"D:\2023-Time\Data+Code\01\UDA-Code\052AdvSKM-CtoK.xlsx"

# 读取Excel文件
df = pd.read_excel(file_path, header=None)

# 选择每隔两行的数据
filtered_df = df.iloc[::3].reset_index(drop=True)

# 创建一个Excel writer对象，以追加模式写入数据到原文件
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
    # 将筛选后的数据保存到"参数"Sheet中
    filtered_df.to_excel(writer, sheet_name="参数", index=False, header=False)

# 读取Excel文件
df = pd.read_excel(file_path, header=None, skiprows=1)

# 选择每隔两行的数据
filtered_df = df.iloc[::3].reset_index(drop=True)

# 创建一个Excel writer对象，以追加模式写入数据到原文件
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
    # 将筛选后的数据保存到"参数"Sheet中
    filtered_df.to_excel(writer, sheet_name="结果", index=False, header=False)

# 读取Excel文件
df = pd.read_excel(file_path, header=None, skiprows=2)

# 选择每隔两行的数据
filtered_df = df.iloc[::3].reset_index(drop=True)

# 创建一个Excel writer对象，以追加模式写入数据到原文件
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
    # 将筛选后的数据保存到"参数"Sheet中
    filtered_df.to_excel(writer, sheet_name="风险", index=False, header=False)