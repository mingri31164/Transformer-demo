"""
GPT / Decoder-Only 架构演示
============================

本文件通过多个场景演示 Decoder-Only（GPT）架构的工作原理和特点。

【场景说明】

Decoder-Only 架构的核心任务是"预测下一个词"。
它不需要编码器来理解外部输入，模型自己就是"语言模型"。
适用于：文本生成、对话系统、代码补全等生成式任务。

【演示内容】

1. 基本前向传播：观察掩码如何生效
2. 掩码机制演示：可视化因果掩码矩阵
3. 自回归生成：逐步生成文本（文字接龙）
4. 与 Transformer 的对比：Decoder-Only 的简化设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt import (
    CausalSelfAttention,
    TransformerBlock,
    GPTModel,
)


# ============================================================
# 第一部分：配置参数
# ============================================================
# GPT-2 Small 配置（适合演示）
VOCAB_SIZE = 10000       # 词表大小（本 demo 使用简化的词表）
D_MODEL = 64             # 模型维度（论文中使用 768，这里缩小以便于演示）
NUM_HEADS = 4            # 注意力头数
NUM_LAYERS = 2           # Transformer Block 层数
D_FF = 128               # 前馈网络维度
MAX_SEQ_LEN = 64         # 最大序列长度
DROPOUT = 0.0            # 演示时不使用 dropout


# ============================================================
# 第二部分：演示掩码机制（核心）
# ============================================================
def demo_causal_mask():
    """
    演示因果掩码（Causal Mask）的工作原理

    这是 Decoder-Only 架构最关键的设计：
    掩码自注意力确保每个位置只能看到当前位置及之前的内容，
    保证了"预测下一个词"任务的公平性。
    """
    print("=" * 70)
    print("演示一：因果掩码机制（Causal Mask）")
    print("=" * 70)

    # 创建掩码自注意力
    attn = CausalSelfAttention(d_model=64, num_heads=4)

    # 创建一个序列
    seq_len = 8
    x = torch.randn(1, seq_len, 64)

    # 前向传播（内部自动应用因果掩码）
    with torch.no_grad():
        output = attn(x)

    # 手动创建掩码矩阵来可视化
    mask = torch.tril(torch.ones(seq_len, seq_len))
    print(f"\n因果掩码矩阵（8x8）:")
    print("  (行=目标位置, 列=源位置)")
    print("  True=可以看到, False=被遮住)\n")
    for i, row in enumerate(mask):
        label = "".join(["[█]" if v == 1 else "[ ]" for v in row.tolist()])
        print(f"  位置 {i}: {label}")

    print(f"""
解读：
  - 位置 0 (对角线): 只能看到自己
  - 位置 1: 只能看到位置 0 和位置 1
  - 位置 2: 只能看到位置 0, 1, 2
  - ...
  - 位置 7: 可以看到所有位置 0~7

  这就是为什么它叫"下三角掩码"——
  上三角区域（未来信息）全部被遮住了！
""")


# ============================================================
# 第三部分：演示前向传播
# ============================================================
def demo_forward_pass():
    """
    演示 GPT 的前向传播过程

    目标：给定一个前缀 "I love this"，预测下一个词
    """
    print("=" * 70)
    print("演示二：前向传播——预测下一个词")
    print("=" * 70)

    model = GPTModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    )

    num_params = model.count_parameters()
    print(f"\n模型参数量: {num_params:,}")
    print(f"  - 词嵌入: {VOCAB_SIZE} × {D_MODEL} = {VOCAB_SIZE * D_MODEL:,}")
    print(f"  - 位置嵌入: {MAX_SEQ_LEN} × {D_MODEL} = {MAX_SEQ_LEN * D_MODEL:,}")
    print(f"  - Transformer Block × {NUM_LAYERS}")
    print(f"  - 最终 Linear 映射到词表: {D_MODEL} × {VOCAB_SIZE} = {D_MODEL * VOCAB_SIZE:,}")

    # 创建输入序列
    # 假设词表: [42, 105, 18, 892] = "I love this movie"
    input_ids = torch.tensor([[42, 105, 18, 892]])

    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, return_logits=True)
        log_probs = model(input_ids, return_logits=False)

    print(f"\n输入序列: [42, 105, 18, 892]  (对应 'I love this movie')")
    print(f"输入形状: {input_ids.shape}")
    print(f"输出 logits 形状: {logits.shape}  (batch_size, seq_len, vocab_size)")
    print(f"         = (1, 4, {VOCAB_SIZE})")
    print(f"         每个位置对词表中每个词的预测得分")

    # 取最后一个位置的 logits（用于预测下一个词）
    last_token_logits = logits[0, -1, :]
    top_k = 5
    top_probs, top_indices = torch.topk(torch.softmax(last_token_logits, dim=-1), k=top_k)

    print(f"\n最后一个位置的预测结果（前 {top_k} 个最可能的词）:")
    print(f"{'排名':<6} {'词索引':<10} {'概率':<12}")
    print("-" * 30)
    for i in range(top_k):
        print(f"  {i+1:<5} {top_indices[i].item():<10} {top_probs[i].item():.4f}")

    print(f"\n解释: 给定前缀 'I love this'，")
    print(f"      模型预测下一个词是索引 {top_indices[0].item()} (概率 {top_probs[0].item():.2%})")


# ============================================================
# 第四部分：演示自回归生成（文字接龙）
# ============================================================
def demo_autoregressive_generation():
    """
    演示自回归生成（Autoregressive Generation）

    这就是 GPT 能够"写文章"的核心机制：
    1. 给定起始文本
    2. 预测下一个词
    3. 将新词添加到输入末尾
    4. 重复
    """
    print("=" * 70)
    print("演示三：自回归生成（文字接龙）")
    print("=" * 70)

    model = GPTModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    )

    model.eval()

    # 起始文本：词表索引
    # 假设: [200] = "Hello", [300] = "world", [400] = "!"
    start_ids = torch.tensor([[200, 300, 400]])

    print(f"\n起始输入: [200, 300, 400]")
    print(f"          (对应 'Hello world !')")
    print(f"\n开始生成 (最多生成 20 个新词)...\n")

    # 使用贪婪采样（temperature=1, top_k=1）
    print(f"{'步骤':<8} {'生成':<15} {'说明':<30}")
    print("-" * 55)

    generated = start_ids.clone()
    max_new_tokens = 10

    with torch.no_grad():
        for step in range(max_new_tokens):
            # 前向传播
            logits = model(generated, return_logits=True)
            next_token_logits = logits[0, -1, :]

            # 贪婪采样：选概率最高的词
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs).unsqueeze(0).unsqueeze(0)

            token_id = next_token.item()

            # 将新词添加到序列
            generated = torch.cat([generated, next_token], dim=1)

            # 显示生成过程
            step_desc = ""
            if step == 0:
                step_desc = "基于 'Hello world !'"
            elif step < 3:
                step_desc = "继续补充..."
            else:
                step_desc = "..." if step < max_new_tokens - 1 else "生成完成"

            print(f"  {step+1:<7} 词索引: {token_id:<10} {step_desc}")

    print(f"\n完整生成的序列长度: {generated.shape[1]} 个词")

    # 展示不同的采样策略
    print(f"\n--- 不同采样策略对比 ---")
    for temp in [0.5, 1.0, 2.0]:
        for tk in [1, 5]:
            gen = model.generate(start_ids, max_new_tokens=5, temperature=temp, top_k=tk)
            tokens = gen[0, len(start_ids[0]):].tolist()
            print(f"  temperature={temp}, top_k={tk} -> {tokens}")


# ============================================================
# 第五部分：Transformer vs Decoder-Only 对比
# ============================================================
def demo_architecture_comparison():
    """
    对比标准 Transformer（Encoder-Decoder）和 Decoder-Only（GPT）
    """
    print("=" * 70)
    print("演示四：Transformer vs Decoder-Only 架构对比")
    print("=" * 70)

    # ----- Encoder Layer (from transformer package) -----
    from transformer import EncoderLayer

    # ----- Decoder Layer (from transformer package) -----
    from transformer import DecoderLayer

    # ----- Decoder Block (from gpt package) -----
    gpt_block = TransformerBlock(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=0.0,
    )

    encoder_layer = EncoderLayer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=0.0,
    )

    decoder_layer = DecoderLayer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=0.0,
    )

    enc_params = sum(p.numel() for p in encoder_layer.parameters())
    dec_params = sum(p.numel() for p in decoder_layer.parameters())
    gpt_params = sum(p.numel() for p in gpt_block.parameters())

    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│              标准 Transformer (Encoder-Decoder)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐     ┌──────────────────┐                 │
│  │   Encoder Layer  │     │   Decoder Layer  │                 │
│  │                  │     │                  │                 │
│  │ [自注意力]  ←─────┼──┐  │ [掩码自注意力] ──┼──┐              │
│  │        ↓         │  │  │        ↓         │  │              │
│  │ [前馈网络]        │  │  │ [交叉注意力] ←───┼──┼──→ Enc 输出   │
│  │        ↓         │  │  │        ↓         │  │              │
│  │ [层归一化]        │  │  │ [前馈网络]        │  │              │
│  └──────────────────┘  │  │        ↓         │  │              │
│         ↓              │  │ [层归一化]        │  │              │
│   编码器输出            │  │        ↓         │  │              │
│                        │  │  解码器输出        │  │              │
│                        └──┴──────────────────┘                 │
│                                                                 │
│  适用场景: 机器翻译、文本摘要、问答等 Seq2Seq 任务                   │
│  参数量: Encoder Layer = {enc_params:,}                          │
│          Decoder Layer = {dec_params:,}                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Decoder-Only (GPT)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐                                           │
│  │  Decoder Block   │  ← 只保留解码器的核心结构                   │
│  │                  │     去掉交叉注意力（不需要编码器输出）        │
│  │ [掩码自注意力] ──┼──┐                                        │
│  │        ↓         │  │                                        │
│  │ [前馈网络]        │  │                                        │
│  │        ↓         │  │                                        │
│  │ [层归一化]        │  │                                        │
│  └────────┬─────────┘  │                                        │
│           ↓              │  (多层堆叠: × N)                       │
│      块输出 ──────────────┘                                        │
│           ↓                                                    │
│     LayerNorm                                                     │
│           ↓                                                    │
│    Linear (→ 词表)                                               │
│                                                                 │
│  适用场景: 文本生成、对话、代码补全等生成任务                        │
│  参数量: Decoder Block = {gpt_params:,}                          │
│                                                                 │
│  优势:                                                            │
│    1. 结构更简单，组件更少，易于规模化                              │
│    2. 训练目标统一（预测下一个词），适合海量无标注文本预训练          │
│    3. 自回归特性天然匹配所有生成式任务                              │
└─────────────────────────────────────────────────────────────────┘
""")


# ============================================================
# 第六部分：逐层观察信息流动
# ============================================================
def demo_layer_progression():
    """
    演示数据通过多层 Decoder Block 时的变化

    观察：每经过一层，模型对上下文的理解是如何深化的
    """
    print("=" * 70)
    print("演示五：多层 Decoder Block 的语义深化过程")
    print("=" * 70)

    model = GPTModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
    )

    # 输入序列
    input_ids = torch.tensor([[200, 105, 18, 892, 42, 105]])
    # 假设对应: "Hello I love this movie I love"

    print(f"\n输入序列: {input_ids[0].tolist()}")
    print(f"序列长度: {input_ids.shape[1]} 个词")
    print(f"模型维度: {D_MODEL}")
    print()

    # 逐层提取中间输出
    model.eval()
    with torch.no_grad():
        # 词嵌入
        token_embeds = model.token_embedding(input_ids)
        print(f"  [词嵌入层]   形状: {token_embeds.shape}")
        print(f"              内容: 每个词是 {D_MODEL} 维的原始语义向量")

        # 位置嵌入
        pos_embeds = model.position_embedding(
            torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        )
        hidden = model.dropout(token_embeds + pos_embeds)
        print(f"\n  [融合后]     形状: {hidden.shape}")
        print(f"              内容: 词嵌入 + 位置嵌入")

        # 通过每一层
        for i, block in enumerate(model.blocks):
            hidden = block(hidden)
            print(f"\n  [Decoder Block {i+1}] 形状: {hidden.shape}")
            print(f"                  内容: 上下文增强后的词向量")
            print(f"                  变化: 向量中融入了其他词的信息")

        # 最终层归一化
        hidden = model.ln_f(hidden)
        print(f"\n  [LayerNorm]   形状: {hidden.shape}")
        print(f"              内容: 数值稳定化后的向量")

        # 语言模型头
        logits = model.lm_head(hidden)
        print(f"\n  [Linear LM头] 形状: {logits.shape}")
        print(f"              内容: 每个词位置对词表中每个词的预测得分")
        print(f"              意义: 取最后一个位置 = 预测下一个词")

    print(f"""
逐层分析：
  Layer 0 (词嵌入):  每个词只知道自己的语义（查字典）
  Layer 1 (Block 1): 每个词开始了解附近的邻居（局部上下文）
  Layer 2 (Block 2): 每个词了解了更远的依赖关系
  ...
  Layer N:           每个词理解了完整的句子语义

这就像阅读理解：
  - 第一遍：认识每个字
  - 第二遍：理解每句话
  - 第三遍：把握整体思想
  层数越多，理解越深刻！
""")


# ============================================================
# 主函数
# ============================================================
def demo():
    """运行所有演示"""
    demo_causal_mask()
    print("\n")
    demo_forward_pass()
    print("\n")
    demo_autoregressive_generation()
    print("\n")
    demo_architecture_comparison()
    print("\n")
    demo_layer_progression()
    print("=" * 70)
    print("演示结束！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
