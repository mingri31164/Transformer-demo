"""
编码器层模块
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """
    Transformer 编码器层
    每个编码器层包含两个子层：
    1. 多头自注意力机制
    2. 位置前馈网络
    每个子层都使用残差连接和层归一化进行包裹。
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # 残差连接与层归一化将在 3.1.2.4 节中详细解释
        # 1. 多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
