"""
Transformer 架构示例
展示如何使用各模块构建完整的 Transformer 模型
"""

import torch
from transformer import (
    PositionalEncoding,
    EncoderLayer,
    DecoderLayer,
)


# ==================== 配置参数 ====================
d_model = 512       # 词嵌入维度
num_heads = 8       # 注意力头数
d_ff = 2048         # 前馈网络隐藏层维度
num_layers = 6      # 编码器/解码器层数
dropout = 0.1       # Dropout 概率


# ==================== 使用示例 ====================
def demo():
    """演示 Transformer 各模块的用法"""
    batch_size = 2
    seq_len = 10

    # 创建编码器层
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)

    # 创建解码器层
    decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

    # 创建位置编码
    pos_encoding = PositionalEncoding(d_model, dropout)

    # 模拟输入
    encoder_input = torch.randn(batch_size, seq_len, d_model)
    decoder_input = torch.randn(batch_size, seq_len, d_model)

    # 应用位置编码
    encoder_input = pos_encoding(encoder_input)
    decoder_input = pos_encoding(decoder_input)

    # 编码器前向传播
    encoder_output = encoder_layer(encoder_input)
    print(f"编码器输出形状: {encoder_output.shape}")

    # 解码器前向传播
    decoder_output = decoder_layer(decoder_input, encoder_output)
    print(f"解码器输出形状: {decoder_output.shape}")


if __name__ == "__main__":
    demo()
