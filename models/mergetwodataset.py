import concurrent.futures
import os
import sys
# 获取当前脚本所在的目录
curPath = os.path.abspath(os.path.dirname(__file__))

# 找到项目根目录
rootPath = os.path.split(curPath)[0]

# 将项目根目录插入到 sys.path 的第一个位置
sys.path.insert(0, rootPath)

def read_file(file_path):
    """
    读取文件内容并返回。

    :param file_path: 文件路径
    :return: 文件内容字符串
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def merge_files(file1, file2, output_file):
    """
    将两个文件的内容合并，并写入输出文件。

    :param file1: 第一个文件路径
    :param file2: 第二个文件路径
    :param output_file: 输出文件路径
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 使用多核读取文件
        future1 = executor.submit(read_file, file1)
        future2 = executor.submit(read_file, file2)

        # 获取文件内容
        content1 = future1.result()
        content2 = future2.result()

    # 合并文件内容
    combined_content = content1 + "\n" + content2

    # 将合并后的内容写入输出文件
    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(combined_content)


if __name__ == "__main__":
    # 输入文件路径
    file1 = "../data/mxm_dataset_train.txt"
    file2 = "../data/mxm_dataset_test.txt"
    # 输出文件路径
    output_file = "../data/mxm_dataset_combined.txt"

    # 调用合并函数
    merge_files(file1, file2, output_file)

    print(f"文件 {file1} 和 {file2} 已成功合并为 {output_file}")
