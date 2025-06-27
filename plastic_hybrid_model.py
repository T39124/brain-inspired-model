import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlasticMultiHeadAttention(nn.Module):
    """可塑性多头注意力层，包含Hebbian学习规则"""

    def __init__(self, embed_dim, num_heads, plasticity_lambda=0.1):
        """
        Args:
            embed_dim: 输入嵌入维度
            num_heads: 注意力头数
            plasticity_lambda: Hebbian可塑性强度系数(0~1)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.plasticity_lambda = plasticity_lambda

        # 验证维度可被头数整除
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        # QKV投影层
        self.W_qkv = nn.Linear(embed_dim, 3 * embed_dim)

        # Hebbian可塑性跟踪矩阵 (每个头独立)
        self.hebbian_trace = nn.Parameter(
            torch.eye(self.head_dim).repeat(num_heads, 1, 1)  # 初始化为单位矩阵
        )

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        # 生成QKV并分割
        qkv = self.W_qkv(hidden_states)  # [batch, seq_len, 3*embed_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # 各[batch, seq_len, embed_dim]

        # 重塑为多头格式 [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, heads, seq_len, seq_len]

        # 应用注意力权重到值向量
        weighted_v = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]

        # Hebbian更新规则 (仅在训练模式更新)
        if self.training:
            # 计算当前批量的Hebbian更新量
            with torch.no_grad():
                # 计算所有位置的外积和
                outer_products = torch.einsum(
                    'bhli,bhlj->bhij',
                    v,
                    weighted_v
                )  # [batch, heads, head_dim, head_dim]

                # 批量平均
                batch_update = outer_products.mean(dim=0)  # [heads, head_dim, head_dim]

                # 应用可塑性规则: 指数移动平均
                self.hebbian_trace.data = (
                        (1 - self.plasticity_lambda) * self.hebbian_trace.data +
                        self.plasticity_lambda * batch_update
                )

        # 应用可塑性变换
        v_transformed = torch.einsum(
            'blhd,hdd->blhd',  # 公式描述：对 head_dim (d) 做矩阵乘法，其他维度保持
            weighted_v.transpose(1, 2),  # [batch, seq_len, heads, head_dim]
            self.hebbian_trace  # [heads, head_dim, head_dim]
        )

        # 合并头输出
        output = v_transformed.reshape(batch_size, seq_len, self.embed_dim)
        return output


class PlasticHybridModel(nn.Module):
    """类脑神经网络混合模型: LSTM + 可塑性多头注意力"""

    def __init__(self, rnn_hidden_size=256, num_heads=4, input_size=128):
        """
        Args:
            rnn_hidden_size: LSTM隐藏层大小
            num_heads: 注意力头数
            input_size: 输入特征维度
        """
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            batch_first=True,
            bidirectional=False
        )
        self.attention = PlasticMultiHeadAttention(
            embed_dim=rnn_hidden_size,
            num_heads=num_heads,
            plasticity_lambda=0.1
        )

    def forward(self, x):
        # LSTM时序处理
        rnn_out, _ = self.rnn(x)  # [batch, seq_len, rnn_hidden_size]

        # 可塑性注意力处理
        plastic_out = self.attention(rnn_out)
        return plastic_out  # [batch, seq_len, rnn_hidden_size]


# 测试示例
if __name__ == "__main__":
    # 创建测试数据
    batch_size, seq_len, features = 8, 50, 128
    test_input = torch.randn(batch_size, seq_len, features)

    # 初始化模型
    model = PlasticHybridModel(
        rnn_hidden_size=256,
        num_heads=4,
        input_size=features
    )

    # 前向传播测试
    output = model(test_input)
    print("输入形状:", test_input.shape)
    print("输出形状:", output.shape)
    print("可塑性跟踪矩阵形状:", model.attention.hebbian_trace.shape)

    # 训练模式测试
    model.train()
    train_output = model(test_input)
    print("\n训练模式输出形状:", train_output.shape)

