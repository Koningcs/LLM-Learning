# Transformer 架构

<div align="center">
  <img src="../img/transformer.png" alt="Transformer 架构">
  <p>图1: Transformer模型的整体架构图</p>
</div>

## Input Embedding
Transformer 的 Input Embedding 主要包含词嵌入和位置编码部分。

### Token Embedding（词嵌入）
词嵌入的过程，就是将一个个词转换为形如 $512 \times 1$ 的向量。具体来说，首先可以把输入的句子视作一个字符串，经过分词器处理后，句子会被切分成一个个词，其中标点符号也会被单独视为一个词，这些词统称为`Token`。
接着，每个`Token`会通过词汇表映射到对应的`Token ID`

![alt text](../img/词嵌入.png)


词汇表是根据分词器训练出来的一个 Token 到 Token ID 的映射结构
```
| Token ID | Token         | 出现频率     |
|----------|---------------|--------------|
| 0        | [PAD]         | 特殊符号     |
| 1        | [UNK]         | 未知词       |
| 2        | [CLS]         | 句子开始     |
| 3        | [SEP]         | 分隔符       |
| 4        | 的            | 很高频       |
| 5        | 是            | 很高频       |
| ...      | ...           | ...          |
| 123      | 我            | 中等频率     |
| 456      | 爱            | 中等频率     |
| 789      | 学习          | 中等
```

假设词汇表是有 10000 个词，那么嵌入矩阵（embedding_matrix）则是$10000 \times 512$的，这样，就能确保每个`Token ID`单独对应一个词嵌入矩阵中的一行，即一个$512 \times 1$ 的向量。
```
  embedding_matrix = [
      [0.1, 0.2, -0.3, ..., 0.5],  # 第0行：ID=0的向量
      [0.4, -0.1, 0.7, ..., 0.2],  # 第1行：ID=1的向量
      ...
      [0.3, 0.8, -0.2, ..., 0.1],  # 第123行：ID=123("我")的向量
      ...
      [0.7, -0.4, 0.9, ..., 0.3],  # 第456行：ID=456("爱")的向量
      ...
      [-0.2, 0.6, 0.1, ..., 0.8],  # 第789行：ID=789("学习")的向量
      ...
  ]
```
嵌入矩阵的参数最初是一些随机数，那他的每一行，为什么能代表一个具体的词呢？这是因为词嵌入矩阵也跟随模型一起训练，一开始的随机数确实不能代表任何语义，但通过一轮轮的训练之后，这些向量就能在这个模型中代表特定的词了
### Positional Encoding（位置编码）

#### 位置编码的作用

由于 Transformer 完全基于注意力机制，**没有使用循环（RNN）或卷积（CNN）结构**，因此模型本身**无法感知序列中词的顺序信息**。

举例来说：
- "我爱学习" 和 "学习爱我" 这两个句子
- 如果没有位置信息，模型看到的只是四个独立的词向量，无法区分它们的顺序

所以必须通过**位置编码**将位置信息注入到输入中，让模型知道每个词在句子中的位置。

#### 位置编码的具体做法

Transformer 使用**正弦和余弦函数**生成位置编码：

**公式：**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- `pos` 是位置索引（第几个词），范围：0, 1, 2, 3, ...
- `i` 是维度索引，范围：0 到 255（因为 $d_{model}=512$）
- `2i` 表示偶数维度使用 sin 函数
- `2i+1` 表示奇数维度使用 cos 函数

**具体例子：**

假设 $d_{model}=512$，对于位置 `pos=0`（第一个词）：
```
维度 0: PE(0,0) = sin(0/10000^0) = 0
维度 1: PE(0,1) = cos(0/10000^0) = 1
维度 2: PE(0,2) = sin(0/10000^(2/512))
维度 3: PE(0,3) = cos(0/10000^(2/512))
...
```

对于位置 `pos=1`（第二个词），这些值会不同，从而区分不同位置。

#### 如何使用位置编码

位置编码会**直接加到词嵌入向量上**：

```
最终输入 = Token Embedding (512×1) + Positional Encoding (512×1)
```

这样，每个词的表示既包含了词本身的语义信息，也包含了它在句子中的位置信息。

#### 为什么选择 sin/cos 函数？

论文给出了几个理由：

1. **可以表示相对位置**：对于任何固定的偏移 k，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数，这使得模型更容易学习关注相对位置
2. **可以外推到更长序列**：即使训练时没见过某个序列长度，推理时也能处理更长的序列
3. **每个维度的波长形成几何级数**：波长从 $2\pi$ 到 $10000 \cdot 2\pi$，不同维度捕获不同尺度的位置信息

#### 与学习式位置编码的对比

论文还实验了**可学习的位置嵌入**（类似词嵌入那样通过训练学习参数），发现两种方法效果几乎相同。但最终选择 sin/cos 函数是因为：
- 不需要额外训练参数
- 可以处理比训练时更长的序列
- 计算效率更高
## Multi-Head Attention（多头注意力机制）

多头注意力是 Transformer 的核心机制，它让模型能够从不同的角度理解输入序列中词与词之间的关系。

### 什么是 Attention（注意力机制）

在理解多头注意力之前，先理解基础的注意力机制。

**核心思想：** 当模型处理某个词时，注意力机制帮助模型决定应该"关注"句子中的哪些其他词。

**例子：**
```
句子："我爱学习 Transformer"
当处理"学习"这个词时：
- 可能需要关注"我"（谁在学习？）
- 可能需要关注"Transformer"（学习什么？）
- 对"爱"的关注度可能较低
```

注意力机制会为每对词计算一个**注意力分数**，分数越高表示关联性越强。

### Self-Attention（自注意力）的计算过程

Transformer 使用的是 **Self-Attention**（自注意力），即序列中的每个词都会与序列中所有词（包括自己）计算注意力。

#### 1. 三个核心矩阵：Q、K、V

对于每个词的嵌入向量（512×1），通过三个不同的权重矩阵进行线性变换，得到三个向量：

- **Q (Query)**：查询向量 - "我想找什么信息？"
- **K (Key)**：键向量 - "我能提供什么信息？"
- **V (Value)**：值向量 - "我的实际内容是什么？"

**计算方式：**
```
Q = W_Q × X    # (64 × 512) × (512 × 1) = (64 × 1)
K = W_Q × X    # (64 × 512) × (512 × 1) = (64 × 1)
V = W_V × X    # (64 × 512) × (512 × 1) = (64 × 1)
```

其中：
- $W_Q, W_K, W_V$ 是可学习的权重矩阵
- X 是输入的词嵌入向量（512维）
- Q、K、V 的维度通常是 64（即 512/8，因为有8个头）

#### 2. 计算注意力分数

**步骤：**

**第一步：计算相似度**
```
Score = Q · K^T
```
用当前词的 Query 与所有词的 Key 做点积，计算相似度。

**例子：** 对于句子 "我 爱 学习"
```
当处理"学习"时：
Score("学习" 对 "我")    = Q_学习 · K_我
Score("学习" 对 "爱")    = Q_学习 · K_爱
Score("学习" 对 "学习")  = Q_学习 · K_学习
```

**第二步：缩放**
```
Scaled Score = Score / √d_k
```
除以 $\sqrt{d_k}$（这里是 $\sqrt{64} = 8$）进行缩放，防止点积值过大导致 softmax 梯度消失。

**第三步：Softmax 归一化**
```
Attention Weights = Softmax(Scaled Score)
```
将分数转换为概率分布，所有权重和为 1。

**例子：**
```
原始分数：  [2.5,  1.8,  3.2]
Softmax后： [0.3,  0.2,  0.5]  # 总和为1，表示注意力分配比例
```

#### 3. 加权求和得到输出

```
Output = Attention Weights × V
```

用注意力权重对所有词的 Value 向量进行加权求和。

**完整公式：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**直观理解：**
- Q 问："我需要什么信息？"
- K 答："我这里有这些信息"
- 通过 Q·K 计算匹配度
- 用匹配度对 V（实际内容）加权平均
- 得到融合了上下文信息的新表示

### 为什么需要 Multi-Head（多头）

单个注意力头只能从一个角度捕获词之间的关系，但语言是复杂的：

**例子：** "The animal didn't cross the street because it was too tired"
- 需要一个头关注：it → animal（指代关系）
- 需要另一个头关注：tired → didn't cross（因果关系）
- 可能还需要关注：cross → street（动作和对象）

**多头注意力允许模型同时从多个角度理解句子。**

### Multi-Head Attention 的实现

Transformer 使用 **8 个并行的注意力头**（head）。

#### 具体步骤：

**1. 将输入分成多个头**

原始输入是 512 维，分成 8 个头，每个头处理 64 维（512/8 = 64）

```
每个头有自己独立的 W_Q, W_K, W_V 参数
Head 1: Q₁, K₁, V₁  (各 64 维)
Head 2: Q₂, K₂, V₂  (各 64 维)
...
Head 8: Q₈, K₈, V₈  (各 64 维)
```

**2. 每个头独立计算注意力**

```
head_i = Attention(Q_i, K_i, V_i)  # 每个头输出 64 维
```

8 个头并行计算，互不干扰。

**3. 拼接所有头的输出**

```
MultiHead Output = Concat(head₁, head₂, ..., head₈)
# 8个头 × 64维 = 512维
```

**4. 线性变换**

```
Final Output = W_O × MultiHead Output
```

通过一个输出权重矩阵 $W_O$（512×512）进行最终的线性变换。

**完整公式：**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中：

$$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

### 参数量计算

以标准 Transformer 为例（$d_{model}=512$，8个头，每个头 $d_k=64$）：

```
每个头的参数：
- W_Q: 512 × 64
- W_K: 512 × 64
- W_V: 512 × 64

8个头总参数：3 × 512 × 64 × 8 = 786,432

输出矩阵 W_O: 512 × 512 = 262,144

总参数量：约 105万 个参数
```

### Multi-Head Attention 的优势

1. **多角度理解**：不同的头可以关注不同类型的关系（语法、语义、指代等）
2. **增强表达能力**：8个头提供了8个不同的表示子空间
3. **并行计算**：所有头可以并行计算，提高效率
4. **降低维度**：每个头只处理 64 维，计算更高效，同时避免过拟合

### 可视化理解

想象你在读一个句子：
- **Head 1** 可能关注语法结构（主谓宾关系）
- **Head 2** 可能关注语义相关性（同义词、反义词）
- **Head 3** 可能关注指代关系（代词指向谁）
- **Head 4** 可能关注位置信息（相邻词的关系）
- ...

最后将这8个不同视角的理解拼接起来，形成对句子更全面的理解。

### 代码示意（伪代码）

```python
class MultiHeadAttention:
    def __init__(self, d_model=512, num_heads=8):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 64

        # 每个头的 Q, K, V 权重矩阵
        self.W_Q = [Matrix(d_model, d_k) for _ in range(num_heads)]
        self.W_K = [Matrix(d_model, d_k) for _ in range(num_heads)]
        self.W_V = [Matrix(d_model, d_k) for _ in range(num_heads)]

        # 输出权重矩阵
        self.W_O = Matrix(d_model, d_model)

    def forward(self, X):
        heads = []
        for i in range(self.num_heads):
            Q = X @ self.W_Q[i]  # (seq_len, d_k)
            K = X @ self.W_K[i]
            V = X @ self.W_V[i]

            # 计算注意力
            scores = Q @ K.T / sqrt(self.d_k)
            attention_weights = softmax(scores)
            head_output = attention_weights @ V

            heads.append(head_output)

        # 拼接所有头
        multi_head = concat(heads)  # (seq_len, d_model)

        # 最终线性变换
        output = multi_head @ self.W_O
        return output
```

## Add & Norm

##