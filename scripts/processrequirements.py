import re

# 读取 requirements.txt 文件
with open('../requirements.txt', 'r', encoding='latin-1') as f:
    lines = f.readlines()

# 用于存储修正后的依赖项
fixed_lines = []

# 逐行处理
for line in lines:
    # 使用正则表达式去除路径部分，只保留包名和版本号
    fixed_line = re.sub(r'\s*@\s*file:///.*', '', line)
    fixed_lines.append(fixed_line)

# 将修正后的内容写回 requirements.txt 文件
with open('requirements.txt', 'w') as f:
    f.writelines(fixed_lines)

print("requirements.txt 文件已修正。")
