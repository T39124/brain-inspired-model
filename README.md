# 混合模型
这个模型纯是想把有趣的东西分享给别人而写出来，但本人并没有任何的代码基础，所以只能使用ai把我的想法实现。

模型是模仿神经元的突触，在Transformer中混合RNN的架构

PlasticMultiHeadAttention (可塑性多头注意力)：

  使用Hebbian学习规则动态更新权重矩阵

  hebbian_trace矩阵存储每个注意力头的可塑性权重

  前向传播时计算注意力权重并更新Hebbian矩阵

  应用可塑性变换增强神经可塑性

  PlasticHybridModel (混合模型)：

  LSTM层：处理时序数据，提取局部特征

  可塑性注意力层：整合全局信息，应用Hebbian规则

  输出维度与LSTM隐藏层大小一致

关键特性：

  Hebbian更新仅在训练模式进行

  使用外积(einsum)计算突触可塑性更新

  可塑性强度通过plasticity_lambda控制

  注意力头独立维护可塑性矩阵
