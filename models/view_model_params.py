import torch
from transformers import BertForSequenceClassification

# 加载预训练的RoBERTa分类模型
model = BertForSequenceClassification.from_pretrained('/root/autodl-tmp/DissertationProject/models/bert_base_uncased')

# 将模型加载到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 计算并打印模型的总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {total_params}")

# 打印每个层的参数名称、形状和参数数量
print("\n每层的参数详细信息：")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"层: {name} | 参数形状: {param.size()} | 参数数量: {param.numel()}")
