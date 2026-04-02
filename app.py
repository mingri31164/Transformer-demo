"""
Transformer 架构演示
===================

通过一个真实的情感分析任务场景，逐步演示 Transformer 各模块的作用与数据流动过程。

【场景设定：情感分析】
给定一段英文评论，模型需要判断该评论的情感是 "正面" 还是 "负面"。

示例输入：
    "This movie is amazing and I really enjoyed it!"
    -> 正面 (Positive)

    "The film was terrible, boring and a waste of time."
    -> 负面 (Negative)

Transformer 在这个任务中的工作流程：
    1. Token Embedding    : 将单词转换为词向量
    2. Positional Encoding: 为词向量添加位置信息
    3. Encoder Stack      : 多层编码器捕获序列内部关系（哪些词与哪些词相关）
    4. Pooling            : 将序列级表示汇聚为单句向量
    5. Linear + Softmax    : 分类输出
"""

import torch
import torch.nn as nn
import math

# ============================================================
# 从 transformer 包中导入所有组件
# ============================================================
from transformer import (
    PositionalEncoding,
    EncoderLayer,
    DecoderLayer,
)


# ============================================================
# 第一部分：配置参数
# ============================================================
# 以下参数是 Transformer 的核心超参数，在真实项目中需要通过实验调优

d_model = 64          # 词嵌入维度（将每个词映射为多维向量）
                     # 维度越高，表达能力越强，但计算量越大
                     # 经典 Transformer 论文中使用 512

num_heads = 4         # 注意力头数
                     # 将 d_model 分成多个子空间并行计算注意力
                     # 每个头可以关注不同方面的关系
                     # 经典值为 8

d_ff = 128            # 前馈网络（Feed-Forward）的隐藏层维度
                     # 通常是 d_model 的 2-4 倍
                     # 经典值为 2048（d_model=512 时）

num_layers = 2        # 编码器/解码器堆叠的层数
                     # 层数越多，模型越深，能捕获更复杂的模式
                     # 经典 BERT/GPT 使用 6-12 层

dropout = 0.1         # Dropout 概率，防止过拟合

vocab_size = 10000   # 词表大小（本 demo 中简化为 10000 个词）


# ============================================================
# 第二部分：模拟数据
# ============================================================
def create_sample_data():
    """
    创建演示用的样本数据

    真实场景中，数据需要经过以下预处理流程：
    1. 分词（Tokenization）：将句子切分为单词/子词
    2. 词表映射（Vocab Lookup）：将每个词转换为词表中的索引
    3. Padding：对齐不同长度的句子
    4. Masking：生成注意力掩码

    这里我们直接创建已经预处理好的数据：
    """
    # 假设我们已经将句子转换为词表索引
    # 每个数字代表词表中一个词的索引
    #
    # 句子 1: "This movie is amazing"  -> [42, 105, 18, 892]
    # 句子 2: "The film was terrible"  -> [7, 389, 156, 2051]

    # [batch_size, seq_len] = [2, 4]
    input_ids = torch.tensor([
        [42, 105, 18, 892],    # 正面评论的词索引
        [7, 389, 156, 2051],  # 负面评论的词索引
    ])

    # 标签：0=负面，1=正面
    labels = torch.tensor([1, 0])

    return input_ids, labels


# ============================================================
# 第三部分：完整的 Transformer 编码器模型
# ============================================================
class TransformerEncoder(nn.Module):
    """
    Transformer 编码器模型（用于情感分析）

    网络结构：
    ┌─────────────────────────────────────────────────┐
    │  Input IDs (词表索引)                             │
    │    ↓                                              │
    │  Embedding (词嵌入层)  — 词表索引 → d_model 维向量   │
    │    ↓                                              │
    │  Positional Encoding (位置编码)                   │
    │    ↓                                              │
    │  × N 个 Encoder Layer (编码器层堆叠)               │
    │    ├── Multi-Head Self-Attention                 │
    │    │     作用：让每个词看到序列中的其他词，          │
    │    │          理解上下文关系（如 "not good" 的组合） │
    │    │     输出：上下文增强的词向量                   │
    │    │                                              │
    │    └── Feed-Forward Network                       │
    │          作用：对每个位置的向量做非线性变换         │
    │          输出：变换后的上下文向量                   │
    │    ↓                                              │
    │  Pooling (汇聚层) — 将序列级表示聚为单句向量         │
    │    ↓                                              │
    │  Linear (线性分类层)                               │
    │    ↓                                              │
    │  Sigmoid (输出 0~1 之间的概率)                     │
    └─────────────────────────────────────────────────┘
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.d_model = d_model

        # 1. 词嵌入层：将词表索引映射为 d_model 维的密集向量
        # 输入: (batch_size, seq_len)  索引值
        # 输出: (batch_size, seq_len, d_model)  词向量
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. 位置编码：为词向量添加位置信息
        # Transformer 本身不感知词的顺序，所以需要显式注入位置信息
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # 3. 编码器层堆叠：N 个连续的编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 4. 分类器
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)  # 输出 1 个 logit
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        """
        前向传播

        Args:
            x: 输入序列的词表索引，(batch_size, seq_len)
            mask: 注意力掩码，可选，用于遮盖 Padding 位置

        Returns:
            情感分类概率，(batch_size,)
        """
        # ----- 阶段 1: 词嵌入 -----
        # "This movie is amazing" 中的 "amazing" 原本只是一个索引 892
        # 嵌入后变成一个 64 维的向量，代表这个词的语义
        #
        # 嵌入后的向量包含信息：
        #   - 词的语义："amazing" 编码为正面情感的词向量
        #   - 但还没有上下文信息！
        x = self.embedding(x)  # (batch_size, seq_len, d_model)

        # ----- 阶段 2: 注入位置编码 -----
        # 词嵌入只包含语义，不包含位置信息
        # 例如 "not good" 和 "good not" 嵌入后看起来相似
        # 位置编码通过正弦/余弦函数为每个位置添加独特信号
        #
        # 位置编码公式：
        #   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        #   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        #
        # 这样每个位置的向量都是独一无二的：
        #   位置 0: [sin(0), cos(0), sin(0/10000), ...]
        #   位置 1: [sin(1), cos(1), sin(1/10000), ...]
        x = self.pos_encoding(x)  # (batch_size, seq_len, d_model)

        # ----- 阶段 3: 通过编码器层 -----
        # 编码器层是 Transformer 的核心，每层包含两个子层：
        #
        # 子层 1 - 多头自注意力（Self-Attention）：
        #   问：这句话中哪些词对理解当前位置最重要？
        #   例子：分析 "amazing" 时，注意力机制可能发现：
        #     "amazing" -> "movie" (主语关系)
        #     "amazing" -> "This"  (情感强度)
        #     "amazing" -> "really"(程度修饰)
        #
        #   Q, K, V 都是从同一个输入 x 得到的（自注意力的 "self" 含义）
        #   注意力分数 = softmax(Q × K^T / √d_k) × V
        #
        # 子层 2 - 前馈网络（Feed-Forward）：
        #   对每个位置的向量独立做非线性变换
        #   作用：进一步处理和提炼来自注意力层的信息
        #
        # 每个子层周围都有残差连接（Residual Connection）和层归一化（Layer Norm）
        # 残差连接：帮助梯度流动，允许构建更深的网络
        # 层归一化：稳定训练，加速收敛
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)  # (batch_size, seq_len, d_model)

        # ----- 阶段 4: 池化 -----
        # 将序列级表示（每个词一个向量）汇聚为单个句子向量
        # 常用策略：
        #   - [CLS] token（BERT 风格）：在序列开头添加特殊分类 token
        #   - Mean Pooling：取所有词向量的平均值
        #   - Max Pooling：取每个维度的最大值
        #
        # 这里使用 Mean Pooling：
        #   将 (batch_size, seq_len, d_model) 沿 seq_len 维度取平均
        #   得到 (batch_size, d_model) — 每句话一个向量
        x = x.mean(dim=1)  # (batch_size, d_model)

        # ----- 阶段 5: 分类 -----
        x = self.dropout(x)
        x = self.fc(x)      # (batch_size, 1)
        x = self.sigmoid(x) # (batch_size, 1)，输出 0~1 之间的概率
        return x.squeeze(-1)  # (batch_size,)


# ============================================================
# 第四部分：演示主函数
# ============================================================
def demo():
    """
    完整演示流程
    """
    print("=" * 70)
    print("Transformer 架构演示 - 情感分析任务")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 步骤 1: 创建模型
    # -----------------------------------------------------------------
    print("\n【步骤 1】创建模型")
    print("-" * 50)

    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
    )

    # 打印模型结构
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    print(f"  - 词嵌入维度: {d_model}")
    print(f"  - 注意力头数: {num_heads}")
    print(f"  - 编码器层数: {num_layers}")
    print(f"  - 前馈网络维度: {d_ff}")

    # -----------------------------------------------------------------
    # 步骤 2: 准备数据
    # -----------------------------------------------------------------
    print("\n【步骤 2】准备数据")
    print("-" * 50)

    input_ids, labels = create_sample_data()
    print(f"输入数据形状: {input_ids.shape}  (batch_size=2, seq_len=4)")
    print(f"  样本 1 (索引): {input_ids[0].tolist()} -> 正面评论")
    print(f"  样本 2 (索引): {input_ids[1].tolist()} -> 负面评论")
    print(f"标签: {labels.tolist()}  (1=正面, 0=负面)")

    # -----------------------------------------------------------------
    # 步骤 3: 前向传播
    # -----------------------------------------------------------------
    print("\n【步骤 3】前向传播")
    print("-" * 50)

    # 设置为评估模式（禁用 dropout）
    model.eval()

    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids)

    print(f"模型输出（情感概率）:")
    print(f"  样本 1: {outputs[0].item():.4f}  (真实标签: 正面)")
    print(f"  样本 2: {outputs[1].item():.4f}  (真实标签: 负面)")

    predicted = (outputs > 0.5).long()
    accuracy = (predicted == labels).float().mean()
    print(f"\n预测结果: {predicted.tolist()}")
    print(f"准确率: {accuracy.item():.2%}")

    # -----------------------------------------------------------------
    # 步骤 4: 分步演示（展示中间输出）
    # -----------------------------------------------------------------
    print("\n【步骤 4】分步演示 - 观察数据在 Transformer 中的变化")
    print("-" * 50)

    # 为了更好地演示，我们只看一个样本
    single_input = input_ids[0:1, :]  # (1, 4)
    print(f"\n以样本 1 为例: 词索引 {single_input[0].tolist()}")
    print("（假设 [42, 105, 18, 892] 对应 'This', 'movie', 'is', 'amazing'）\n")

    # 创建位置编码（单独演示）
    pos_enc = PositionalEncoding(d_model, dropout)

    # ---- 子步骤 4.1: 词嵌入 ----
    embedding_layer = nn.Embedding(vocab_size, d_model)
    embedded = embedding_layer(single_input)
    print(f"  4.1 词嵌入后:")
    print(f"      形状: {embedded.shape}")
    print(f"      每个词被编码为一个 {d_model} 维的向量")
    print(f"      'amazing' (索引 892) 的词向量前 5 维:")
    print(f"      {embedded[0, 3, :5].tolist()}")

    # ---- 子步骤 4.2: 位置编码 ----
    with_pos_enc = pos_enc(embedded)
    print(f"\n  4.2 加上位置编码后:")
    print(f"      形状: {with_pos_enc.shape}")
    print(f"      位置编码为每个位置添加独特的信号:")
    print(f"      位置 0 (This) 的编码前 5 维: {pos_enc.pe[0, 0, :5].tolist()}")
    print(f"      位置 3 (amazing) 的编码前 5 维: {pos_enc.pe[0, 3, :5].tolist()}")

    # ---- 子步骤 4.3: 通过编码器层 ----
    print(f"\n  4.3 通过编码器层:")
    for i, enc_layer in enumerate(model.encoder_layers):
        with_pos_enc = enc_layer(with_pos_enc)
        print(f"      第 {i+1} 层编码器后: 形状 {with_pos_enc.shape}")
        print(f"      每个词现在都带有上下文信息了！")

    # ---- 子步骤 4.4: 池化 ----
    pooled = with_pos_enc.mean(dim=1)
    print(f"\n  4.4 Mean Pooling (平均池化):")
    print(f"      形状: {pooled.shape}  (序列信息汇聚为单句向量)")
    print(f"      这个向量包含了整句话的语义信息")

    # ---- 子步骤 4.5: 分类 ----
    logits = model.fc(pooled)
    probs = model.sigmoid(logits)
    print(f"\n  4.5 分类输出:")
    print(f"      线性层输出 (logits): {logits.squeeze().item():.4f}")
    print(f"      Sigmoid 后概率: {probs.squeeze().item():.4f}")
    print(f"      -> {'正面' if probs.item() > 0.5 else '负面'}")

    # -----------------------------------------------------------------
    # 步骤 5: 注意力机制演示
    # -----------------------------------------------------------------
    print("\n【步骤 5】注意力机制可视化")
    print("-" * 50)
    print("""
    注意力机制是 Transformer 的核心创新！

    传统 RNN 必须从头到尾顺序处理序列（无法并行），且难以捕获远距离依赖。
    Transformer 通过自注意力机制：
    [OK] 可以并行处理整个序列（速度更快）
    [OK] 可以直接关注序列中任意两个位置（长距离依赖）

    示例 - 分析 "This movie is really amazing!"
    假设我们要计算 "amazing" 的注意力：

    注意力权重可能类似于：
        关注 "This"    [****----]  0.35
        关注 "movie"   [**********]  0.45 (主语关系最强)
        关注 "is"      [**------]  0.10
        关注 "really"  [*******--]  0.30 (程度修饰)
        关注 "amazing" [******--]  0.25 (自注意力)

    这说明 "amazing" 与 "movie" 关系最密切，
    模型理解了 "什么" 是 "amazing" 的！
    """)

    # -----------------------------------------------------------------
    # 步骤 6: 与其他模型对比
    # -----------------------------------------------------------------
    print("\n【步骤 6】Transformer vs 传统模型")
    print("-" * 50)
    print("""
    ┌──────────────────┬───────────────────────────────────────────────┐
    │      模型          │                      特点                      │
    ├──────────────────┼───────────────────────────────────────────────┤
    │  传统 RNN/LSTM    │  顺序处理，速度慢；长距离依赖弱；难以并行        │
    │  CNN (TextCNN)    │  并行好，但感受野有限，难以捕获全局关系         │
    │  Transformer      │  [OK] 完全并行  [OK] 任意位置直接建模  [OK] 效果好      │
    └──────────────────┴───────────────────────────────────────────────┘

    Transformer 的优势场景：
    [*] 机器翻译：源语言句子和目标语言句子之间的对齐
    [*] 文本生成：GPT 系列基于 Transformer 的解码器
    [*] 预训练模型：BERT、GPT、T5 等都是基于 Transformer
    """)

    print("=" * 70)
    print("演示结束！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
