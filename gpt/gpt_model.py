"""
GPT / Decoder-Only 架构主模型
==============================

GPT (Generative Pre-trained Transformer) 的核心思想：
"语言的核心任务是预测下一个最有可能出现的词。"

Decoder-Only 架构只保留 Transformer 的解码器部分，
通过掩码自注意力实现自回归（Autoregressive）生成。

【架构对比】

标准 Transformer（Encoder-Decoder）：
    编码器: [自注意力] → [交叉注意力] → [FFN]
    解码器: [掩码自注意力] → [交叉注意力] → [FFN]
    用途: 机器翻译、文本摘要等 Seq2Seq 任务

Decoder-Only (GPT):
    解码器: [掩码自注意力] → [FFN] → [掩码自注意力] → [FFN] → ...
    用途: 文本生成、对话、代码补全等生成任务

【GPT 架构特点】

1. 没有编码器：不需要理解外部输入，模型自己就是"世界模型"
2. 掩码自注意力：预测第 t 个词时，只能使用 0~t-1 的信息
3. 层层堆叠：通过多层 Transformer Block 逐层深化理解
4. 统一预训练目标：预测下一个词（Next Token Prediction）
"""

import torch
import torch.nn as nn
import math
from .transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """
    GPT / Decoder-Only 主模型

    网络结构:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input Tokens (词表索引)                                     │
    │    ↓                                                         │
    │  Token Embedding (词嵌入层)                                   │
    │    ↓                                                         │
    │  Positional Embedding / RoPE (位置编码)                       │
    │    ↓                                                         │
    │  × N 个 Transformer Decoder Block                            │
    │    ├── Masked Self-Attention (掩码自注意力)                   │
    │    └── Feed-Forward Network (前馈网络)                        │
    │    ↓                                                         │
    │  × N 个 Transformer Decoder Block ...                        │
    │    ↓                                                         │
    │  LayerNorm                                                   │
    │    ↓                                                         │
    │  Linear (语言模型头)                                          │
    │    ↓                                                         │
    │  输出: 每个词表中每个词的概率分布                               │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        """
        初始化 GPT 模型

        Args:
            vocab_size: 词表大小
            d_model: 模型维度（每个词的向量长度）
            num_heads: 注意力头数
            num_layers: Transformer Block 的层数（即模型深度）
            d_ff: 前馈网络的隐藏层维度
            max_seq_len: 最大序列长度（用于位置编码）
            dropout: Dropout 概率
            padding_idx: 填充标记的索引（用于忽略 Padding 位置）
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx

        # ----- 组件 1: 词嵌入层 -----
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )

        # ----- 组件 2: 位置嵌入 -----
        # GPT-1/GPT-2 使用可学习的位置嵌入
        # GPT-4、LLaMA 等使用 RoPE（旋转位置编码），此处使用可学习位置嵌入作为演示
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # ----- 组件 3: Dropout -----
        self.dropout = nn.Dropout(dropout)

        # ----- 组件 4: 多层 Transformer Block -----
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ----- 组件 5: 最终层归一化 -----
        self.ln_f = nn.LayerNorm(d_model)

        # ----- 组件 6: 语言模型头 -----
        # 将 d_model 维向量映射回词表维度，得到每个词的概率
        # 权重通常与 token_embedding 共享（tied weights），节省约 30% 的参数量
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定：lm_head 与 token_embedding 共享权重
        # 这样词嵌入学到的表示和输出层直接对应，训练更高效
        self.lm_head.weight = self.token_embedding.weight

        # ----- 参数初始化 -----
        self._init_weights()

    def _init_weights(self):
        """参数初始化，使用标准 GPT-2 的初始化策略"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 标准差 = 0.02 或 0.02 / √(2 * num_layers)
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        创建 Padding 掩码

        作用：将输入中的 Padding 位置（通常为 0）遮住，
        防止模型关注无意义的 Padding。

        Args:
            input_ids: 词表索引，(batch_size, seq_len)

        Returns:
            mask: 掩码张量，(batch_size, seq_len)
                  1 表示有效位置，0 表示 Padding 位置
        """
        # 找出 Padding 位置（通常为 padding_idx）
        mask = (input_ids != self.padding_idx).float()
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        前向传播（训练模式）

        Args:
            input_ids: 输入词表索引，(batch_size, seq_len)
            return_logits: 若为 True，返回未经 Softmax 的原始 logits

        Returns:
            logits: 形状 (batch_size, seq_len, vocab_size)
                    表示序列中每个位置对词表中每个词的预测得分
        """
        batch_size, seq_len = input_ids.size()

        # ----- 安全检查：序列长度不能超过最大长度 -----
        assert seq_len <= self.max_seq_len, (
            f"输入序列长度 {seq_len} 超过了最大长度 {self.max_seq_len}。"
            "请在创建模型时增加 max_seq_len 参数。"
        )

        # ----- 步骤 1: 词嵌入 -----
        # 形状: (batch_size, seq_len) → (batch_size, seq_len, d_model)
        token_embeds = self.token_embedding(input_ids)

        # ----- 步骤 2: 位置嵌入 -----
        # 为每个位置添加位置编码
        # 位置索引: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)

        # ----- 步骤 3: 融合词嵌入 + 位置嵌入 -----
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        # ----- 步骤 4: 通过多层 Transformer Block -----
        # 每一层都会深化对上下文的理解
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # ----- 步骤 5: 最终层归一化 -----
        hidden_states = self.ln_f(hidden_states)

        # ----- 步骤 6: 语言模型头 -----
        # 映射到词表维度，得到每个位置对词表中每个词的打分
        logits = self.lm_head(hidden_states)

        if return_logits:
            return logits
        else:
            return torch.log_softmax(logits, dim=-1)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_token_id: int = None,
    ) -> torch.Tensor:
        """
        自回归文本生成

        这个方法是 GPT 能够"写文章"的核心。
        它一次生成一个词，每次生成都把之前生成的词作为新的输入。

        工作原理（自回归 Autoregressive）：
        输入: "The cat sat on the"
        1. 模型预测下一个词: "mat" (概率最高)
        2. 更新输入: "The cat sat on the mat"
        3. 再次预测: "and" ...
        4. 重复直到达到停止条件

        Args:
            input_ids: 初始输入，(batch_size, seq_len)
            max_new_tokens: 最多生成的新词数量
            temperature: 温度参数，控制随机性
                        - temperature → 0: 输出确定，always 选最高概率的词
                        - temperature = 1: 保持原始概率分布
                        - temperature → ∞: 概率分布趋均匀，完全随机
            top_k: Top-K 采样，限制只从概率最高的 k 个词中采样
            top_p: Top-P（核采样），限制从累积概率超过 p 的最小词集合中采样
            stop_token_id: 停止词 ID，生成到此词时停止

        Returns:
            生成的完整序列，(batch_size, seq_len + max_new_tokens)
        """
        self.eval()  # 生成时使用 eval 模式

        with torch.no_grad():
            generated = input_ids.clone()

            for _ in range(max_new_tokens):
                # 如果当前序列长度超过最大长度，进行截断
                # （位置嵌入最大长度为 max_seq_len）
                if generated.size(1) >= self.max_seq_len:
                    generated = generated[:, -self.max_seq_len:]

                # 前向传播得到 logits
                logits = self.forward(generated, return_logits=True)

                # 只取最后一个位置的 logits（预测的下一个词）
                next_token_logits = logits[:, -1, :] / temperature

                # 应用 Top-K 过滤
                if top_k > 0:
                    # 保留概率最高的 top_k 个词
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

                # 应用 Top-P（核采样）
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # 找到累积概率超过 top_p 的位置，从下一个位置开始过滤
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')

                # 从概率分布中采样下一个词
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # 拼接到序列末尾
                generated = torch.cat([generated, next_token], dim=1)

                # 如果生成了停止词，停止
                if stop_token_id is not None and next_token.item() == stop_token_id:
                    break

            return generated

    def count_parameters(self) -> int:
        """统计模型可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GPT1(GPTModel):
    """
    GPT-1 配置 (原生参数)
    - 12 层
    - 768 维
    - 12 个注意力头
    - 3072 维前馈网络
    - 约 1.17 亿参数
    """

    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(
            vocab_size=vocab_size,
            d_model=768,
            num_heads=12,
            num_layers=12,
            d_ff=3072,
            max_seq_len=512,
            dropout=0.1,
            **kwargs,
        )


class GPT2Small(GPTModel):
    """
    GPT-2 Small 配置
    - 12 层
    - 768 维
    - 12 个注意力头
    - 3072 维前馈网络
    - 约 1.24 亿参数
    """

    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(
            vocab_size=vocab_size,
            d_model=768,
            num_heads=12,
            num_layers=12,
            d_ff=3072,
            max_seq_len=1024,
            dropout=0.1,
            **kwargs,
        )
