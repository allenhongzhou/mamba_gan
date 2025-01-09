import pandas as pd

# 读取Excel文件中的原始sheet
input_file = 'a.xlsx'  # 替换为你的输入Excel文件路径
output_file = 'b.xlsx'  # 结果保存为新的Excel文件
sheet_name = 'a'  # 替换为原始sheet的名称

# 读取原始数据
df = pd.read_excel(input_file, sheet_name=sheet_name)

# 提取C1-C5列（假设前5列为C1-C5，其中C5为学校列）
df = df.iloc[:, 1:6]

# 以C5列（学校列）分组
grouped = df.groupby(df.columns[4])  # 第5列为C5

# 创建一个新的Excel writer
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    for school_name, group in grouped:
        # 将每个学校的数据写入一个新的sheet，保留C1-C4列
        group_c1_c4 = group.iloc[:, :5]  # 保留C1-C4列
        group_c1_c4.to_excel(writer, sheet_name=str(school_name), index=False)

print(f"数据已按学校分组并保存到 {output_file} 中。")
