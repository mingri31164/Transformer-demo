"""
Transformer Decoder Block（解码器块）
======================================

Decoder-Only 架构由多层 Decoder Block 堆叠而成。

【Decoder Block vs Encoder Layer 的区别】

Encoder Layer（来自标准 Transformer）：
  输入 → [自注意力(看全部)] → [前馈网络] → 输出

Decoder Block（Decoder-Only / GPT）：
  输入 → [掩码自注意力(只看过去)] → [前馈网络] → 输出

Decoder Block 比 Encoder Layer 少了一个"交叉注意力"子层：
  - 不需要编码器的输出（因为没有编码器）
  - 只需要掩码自注意力（保证自回归特性）
  - 因此结构更简单，更容易规模化

【与 GPT 论文中的名称对应】

GPT-1/GPT-2 论文中，Decoder Block 的结构被称为 "Masked Self-Attention"。
本模块是构建 GPT 模型的基础组件。
"""

import torch
import torch.nn as nn
from .causal_attention import CausalSelfAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer 解码器块（Decoder Block）

    每个 Block 包含两个子层：
    1. Causal Self-Attention（掩码自注意力）
    2. Feed-Forward Network（前馈网络）

    每个子层都包裹在残差连接（Residual Connection）和层归一化（Layer Normalization）中。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        # 子层 1: 掩码自注意力
        self.self_attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
        )

        # 子层 2: 前馈网络
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation="gelu",
        )

        # 两个层归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
            mask: 可选掩码（如 Padding 掩码）

        Returns:
            输出张量，形状 (batch_size, seq_len, d_model)
        """
        # ----- 残差连接 #1: 掩码自注意力 -----
        # 注意力输出与输入相加（残差连接）
        # 再应用层归一化
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # ----- 残差连接 #2: 前馈网络 -----
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class GPTDecoderBlock(TransformerBlock):
    """
    GPT 解码器块的别名，与 TransformerBlock 功能完全相同。
    提供这个别名是为了代码可读性——明确表明这是用于 GPT 架构的。
    """
    pass
