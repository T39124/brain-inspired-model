from plastic_hybrid_model import PlasticHybridModel
import torch

# 初始化模型
model = PlasticHybridModel(
    rnn_hidden_size=512,
    num_heads=8,
    input_size=256
)

# 测试数据
inputs = torch.randn(16, 100, 256)  # [batch, seq_len, features]

# 前向传播
outputs = model(inputs)
print("模型输出形状:", outputs.shape)