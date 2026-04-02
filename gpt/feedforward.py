"""
Decoder-Only 架构的前馈网络
============================

Decoder-Only 架构的前馈网络与标准 Transformer 编码器中完全相同。
它的作用是：对每个位置独立做非线性变换，进一步提炼来自注意力层的信息。

【为什么注意力层之外还需要前馈网络？】

注意力机制本质上是"加权求和"——它重组信息，但不改变向量本身。
如果只用注意力层，无论堆叠多少层，网络都只能做线性变换（矩阵乘法 + Softmax）。
前馈网络通过非线性激活函数（ReLU/GELU），为模型提供了真正的非线性表达能力。

【与编码器中 FFN 的区别】

本模块的设计与编码器中的 PositionWiseFeedForward 完全一致，
但为了保持 gpt/ 包的独立性，在此单独实现。
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    位置前馈网络 (Position-wise Feed-Forward Network)

    结构: Linear(d_model → d_ff) → Activation → Dropout → Linear(d_ff → d_model)

    其中 d_ff 通常是 d_model 的 2~4 倍，提供更大的容量。
    激活函数可以使用 ReLU（GPT-1/2）或 GELU（GPT-3 及以后）。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        # 第一层：扩展维度
        self.linear1 = nn.Linear(d_model, d_ff)

        # 激活函数：GPT-2 使用 GELU，GPT-3+ 也使用 GELU
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout：随机丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 第二层：恢复维度
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)

        Returns:
            输出张量，形状 (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)       # (batch_size, seq_len, d_ff)
        x = self.activation(x)    # 非线性激活
        x = self.dropout(x)        # Dropout
        x = self.linear2(x)        # (batch_size, seq_len, d_model)
        return x
