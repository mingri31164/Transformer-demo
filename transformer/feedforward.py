"""
位置前馈网络模块
"""

import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络模块
    对序列中每个位置独立应用相同的非线性变换，由两个线性层组成，
    中间使用 ReLU 激活函数。
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # 最终输出形状: (batch_size, seq_len, d_model)
        return x
