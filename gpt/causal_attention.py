"""
掩码自注意力 (Causal Self-Attention)
=====================================

这是 Decoder-Only 架构的核心模块。

【与标准自注意力的区别】

标准多头注意力（Encoder/Decoder 中的交叉注意力）：
  - 每个位置可以看到序列中的 ALL 位置（前后的词都能看到）
  - 用于理解输入的全部内容

掩码自注意力（Decoder-Only 独有）：
  - 每个位置只能看到当前位置及之前的所有位置
  - 当前位置之后的词必须被"掩码"遮住（设为 -∞）
  - 这是自回归生成的基础——预测第 t 个词时，不能偷看 t+1, t+2, ...

【为什么必须这样做】

如果不做掩码，解码器在训练时会"作弊"：
  输入: "I love this movie"
  位置:   0  1    2    3    4

  训练目标——预测 "love"：
    如果能看到位置 2 的 "this"，模型可以直接复制答案
    正确做法：只看位置 0 的 "I"，预测下一个词是 "love"

掩码机制保证了训练目标与推理（实际使用）时的一致性。
"""

import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    """
    掩码自注意力模块

    与 MultiHeadAttention 的关键区别：
    1. 注意力计算后会应用一个"下三角"掩码
    2. 掩码将所有未来位置（t+1, t+2, ...）的注意力分数设为 -inf
    3. Softmax 后这些位置的权重变为 0

    这确保了在预测第 t 个词时，模型只能使用位置 0 到 t-1 的信息。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        # Q, K, V 线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
            mask: 可选的下三角掩码，形状 (seq_len, seq_len)
                  mask[t, s] = 1 表示位置 s 可以被位置 t 关注
                  mask[t, s] = 0 表示位置 t 不能看到位置 s

        Returns:
            输出张量，形状 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()

        # ----- 步骤 1: 线性变换得到 Q, K, V -----
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # ----- 步骤 2: 分头 (Multi-head) -----
        # 将 d_model 维度拆分为 num_heads 个头
        # 形状: (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # ----- 步骤 3: 计算注意力分数 (QK^T / √d_k) -----
        # 形状: (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # ----- 步骤 4: 应用因果掩码 (Causal Mask) -----
        # 这是 Decoder-Only 与 Encoder 的关键区别！
        #
        # 创建下三角掩码矩阵（下三角为 1，上三角为 0）：
        #
        #              目标位置 (t)
        #              0    1    2    3    4
        # 源位置(s) 0 [ True  -   -    -    -  ]
        #           1 [ True True  -    -    -  ]
        #           2 [ True True True   -    -  ]
        #           3 [ True True True True   -  ]
        #           4 [ True True True True True ]
        #
        # 对角线为 True（每个位置可以关注自己）
        # 对角线上方为 False（不能看未来的词）
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()

        # 如果外部提供了额外的掩码（如 Padding 掩码），与因果掩码结合
        if mask is not None:
            # mask: (batch_size, 1, seq_len, seq_len) 或 (1, 1, seq_len, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) & mask
        else:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # 将被掩码的位置设为 -inf，Softmax 后权重为 0
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        # ----- 步骤 5: Softmax 得到注意力权重 -----
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # ----- 步骤 6: 加权求和得到输出 -----
        # 形状: (batch_size, num_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_probs, V)

        # ----- 步骤 7: 合并多头 -----
        # 形状: (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)

        # ----- 步骤 8: 最终线性变换 -----
        output = self.W_o(attn_output)
        return output


class MultiHeadCausalAttention(nn.Module):
    """
    多头掩码自注意力的简化封装（替代 CausalSelfAttention 的别名）
    提供更清晰的接口用于自定义注意力机制。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, num_heads, dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_out = self.attention(x, mask)
        return self.dropout(self.proj(attn_out))
