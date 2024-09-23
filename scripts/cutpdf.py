from PyPDF2 import PdfReader, PdfWriter

# 打开原始PDF文件
input_pdf_path = './myReport.pdf'  # 输入PDF文件路径
output_pdf_path = '230123191.pdf'  # 输出新的PDF文件路径
# 打开PDF文件
reader = PdfReader(input_pdf_path)
writer = PdfWriter()

# 选择要提取的页面范围：1-25页（对应PDF中的第0-24页）和63-65页（对应PDF中的第62-64页）
pages_to_extract = list(range(0, 25)) + list(range(58, 62))  # 页码从0开始

# 将选定的页面添加到PDF writer
for page_num in pages_to_extract:
    page = reader.pages[page_num]
    writer.add_page(page)

# 将选定的页面保存为新的PDF文件
with open(output_pdf_path, 'wb') as output_pdf:
    writer.write(output_pdf)

print(f"选定的页面已保存为新的PDF文件：{output_pdf_path}")
