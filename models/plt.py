import matplotlib.pyplot as plt
import os
# 获取脚本所在的目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 设置保存图片的目录（相对路径）
save_dir = os.path.join(base_dir, '.', 'plt')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 示例数据
epochs = [0.5, 1, 2, 3, 4]
training_accuracy = [0.6, 0.7, 0.8, 0.9, 0.95]
validation_accuracy = [0.65, 0.7, 0.75, 0.85, 0.88]
training_loss = [0.7, 0.6, 0.4, 0.3, 0.2]
validation_loss = [0.68, 0.62, 0.6, 0.55, 0.5]

# 创建子图
plt.figure(figsize=(12, 5))

# 绘制训练和验证准确率
plt.subplot(1, 2, 1)
plt.plot(epochs, training_accuracy, label='Training Accuracy')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy of Transformer')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 绘制训练和验证损失
plt.subplot(1, 2, 2)
plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, validation_loss, label='Validation Loss')
plt.title('Training and Validation Loss of Transformer')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(save_dir, 'training_validation_loss.png'), dpi=300)

# 显示图像
plt.show()
