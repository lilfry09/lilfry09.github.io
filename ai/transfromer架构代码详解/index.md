# Transformer架构代码详解


# 1，位置编码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 检查是否有可用的GPU，有就用，没有就用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"我们将使用 {device} 设备进行计算")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        :param d_model: 词嵌入的维度，必须是偶数
        :param max_len: 句子的最大长度
        :param dropout: Dropout比例
        """
        super(PositionalEncoding, self).__init__() # 初始化父类
        assert d_model % 2 == 0
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够大的位置编码矩阵，预先计算好
        pe = torch.zeros(max_len, d_model)
        
        # 创建一个位置张量 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数的分母部分，使用log可以防止数值溢出
        # div_term 的形状是 (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 使用正弦和余弦函数填充位置编码矩阵
        # 偶数维度用sin，奇数维度用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 将pe从 [max_len, d_model] 变为 [1, max_len, d_model] 以便与输入批次兼容
        pe = pe.unsqueeze(0)
        
        # 将pe注册为模型的buffer。它不是模型的参数，但希望它能和模型一起保存和移动（比如.to(device)）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: 输入的词嵌入张量，形状为 [batch_size, seq_len, d_model]
        """
        # 将x与位置编码相加。我们只取序列长度那么多的位置编码
        # self.pe[:, :x.size(1), :] 的形状是 [1, seq_len, d_model]
        # 利用广播机制，它会自动扩展到 [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```
# 2，多头注意力机制

```python   

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1): 
        """
        :param d_model: 词嵌入的维度
        :param nhead: 注意力头的数量
        """
        super().__init__() # 初始化父类
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"

        self.d_model = d_model # 输入和输出的维度
        self.nhead = nhead # 注意力头的数量
        self.d_k = d_model // nhead  # 每个头的维度

        # 定义Q, K, V和输出的线性变换层
        self.w_q = nn.Linear(d_model, d_model) # 查询的线性变换
        self.w_k = nn.Linear(d_model, d_model) # 键的线性变换
        self.w_v = nn.Linear(d_model, d_model) # 值的线性变换
        self.w_o = nn.Linear(d_model, d_model) # 输出的线性变换
        
        self.dropout = nn.Dropout(p=dropout) # Dropout层，防止过拟合

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: 输入张量，形状 [batch_size, seq_len, d_model]
        mask: 掩码，用于屏蔽某些位置的注意力
        """
        batch_size = query.size(0) # 获取批次大小

        # 1. 线性变换，得到Q, K, V
        Q = self.w_q(query) # 查询的线性变换
        K = self.w_k(key)
        V = self.w_v(value)

        # 2. 将Q, K, V拆分成多个头
        # 原始: [batch, len, d_model] -> 拆分: [batch, nhead, len, d_k]
        Q = Q.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # 3. 计算注意力分数
        # scores 的形状: [batch, nhead, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. 应用掩码 (如果提供)
        if mask is not None:
            # mask需要广播到scores的形状
            scores = scores.masked_fill(mask == 0, -1e9) # 用一个很大的负数填充

        # 5. 应用Softmax得到注意力权重
        # attn_weights 的形状: [batch, nhead, len_q, len_k]
        attn_weights = F.softmax(scores, dim=-1) 
        attn_weights = self.dropout(attn_weights) # 应用Dropout

        # 6. 注意力权重乘以V
        # context 的形状: [batch, nhead, len_q, d_k]
        context = torch.matmul(attn_weights, V) 

        # 7. 合并多头的结果
        # context -> [batch, len_q, nhead, d_k] -> [batch, len_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 8. 最后的线性变换
        output = self.w_o(context) 
        return output
```

# 3，前馈神经网络
```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: 输入输出维度
        :param d_ff: 中间层的维度，通常是d_model的4倍
        """
        super(PositionwiseFeedForward, self).__init__() # 初始化父类
        self.linear1 = nn.Linear(d_model, d_ff) # 第一层线性变换
        self.dropout = nn.Dropout(p=dropout) # Dropout层，防止过拟合
        self.linear2 = nn.Linear(d_ff, d_model) # 第二层线性变换，将中间层的输出映射回d_model维度

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.linear1(x)   # 先通过第一层线性变换
        x = F.relu(x) # 应用ReLU激活函数
        x = self.dropout(x) # 应用Dropout
        x = self.linear2(x) # 再通过第二层线性变换
        return x # 输出形状仍然是 [batch_size, seq_len, d_model]
```

# 4，层归一化
```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```
# 5，残差连接和残差归一化
```python
class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # x: [batch_size, seq_len, d_model]
        # sublayer: [batch_size, seq_len, d_model]
        return x + self.dropout(sublayer(self.norm(x)))
```

# 6，编码器层
```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 定义两个层归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, src, src_mask):
        # 1. 多头自注意力 + Add & Norm
        # 残差连接：将输入src与注意力输出相加
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src) # 层归一化

        # 2. 前馈网络 + Add & Norm
        # 残差连接：将上一步的输出与前馈网络输出相加
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src) # 层归一化
        
        return src

```
# 7,解码器层
```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 第一个注意力：带掩码的自注意力
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # 第二个注意力：交叉注意力
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 定义三个层归一化层
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(p=dropout) 
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # 1. 带掩码的多头自注意力 + Add & Norm
        # trg_mask确保解码时只能看到过去的信息
        attn_output = self.self_attn(trg, trg, trg, mask=trg_mask)
        trg = trg + self.dropout1(attn_output)
        trg = self.norm1(trg)

        # 2. 交叉注意力 + Add & Norm
        # Query来自解码器(trg)，Key和Value来自编码器(enc_src)
        # src_mask屏蔽编码器输出中的填充部分
        cross_attn_output = self.cross_attn(trg, enc_src, enc_src, mask=src_mask)
        trg = trg + self.dropout2(cross_attn_output)
        trg = self.norm2(trg)

        # 3. 前馈网络 + Add & Norm
        ff_output = self.feed_forward(trg)
        trg = trg + self.dropout3(ff_output)
        trg = self.norm3(trg)
        
        return trg

```
# 8，Transformer模型
```python

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, d_ff, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()

        # --- 编码器部分 ---
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = EncoderLayer(d_model, nhead, d_ff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # --- 解码器部分 ---
        self.decoder_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_len, dropout)
        decoder_layer = DecoderLayer(d_model, nhead, d_ff, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # --- 输出层 ---
        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """生成一个上三角矩阵的掩码，用于防止decoder看到未来的信息"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        # 创建掩码
        src_padding_mask = (src == 0) # 假设0是PAD_TOKEN
        trg_padding_mask = (trg == 0)
        trg_subsequent_mask = self.generate_square_subsequent_mask(trg.size(1))

        # --- 编码器 ---
        src = self.encoder_embedding(src) # 词嵌入
        src = self.pos_encoder(src) # 添加位置编码
        # 注意：PyTorch内置的TransformerEncoder需要[seq_len, batch_size, d_model]的输入
        src = src.transpose(0, 1)
        memory = self.encoder(src, src_key_padding_mask=src_padding_mask)

        # --- 解码器 ---
        trg = self.decoder_embedding(trg)
        trg = self.pos_decoder(trg)
        # PyTorch内置的TransformerDecoder也需要[seq_len, batch_size, d_model]的输入
        trg = trg.transpose(0, 1)
        output = self.decoder(trg, memory, 
                              tgt_mask=trg_subsequent_mask, 
                              tgt_key_padding_mask=trg_padding_mask,
                              memory_key_padding_mask=src_padding_mask)
        
        # 将输出变回 [batch_size, seq_len, d_model]
        output = output.transpose(0, 1)

        # --- 输出层 ---
        output = self.fc_out(output)
        
        return output
```
# 9，模型训练和评估
```python
import torch.optim as optim
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, d_ff, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        # ... (省略前面的代码)

    def train_model(self, train_loader, criterion, optimizer, num_epochs=10):
        self.train()  # 设置模型为训练模式
        for epoch in range(num_epochs):
            total_loss = 0
            for src, trg in train_loader:
                optimizer.zero_grad()  # 清空梯度
                output = self(src, trg[:, :-1])  # trg[:-1]是因为我们不需要最后一个token
                loss = criterion(output.view(-1, output.size(-1)), trg[:, 1:].contiguous().view(-1))  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                total_loss += loss.item()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

    def evaluate(self, eval_loader):
        self.eval()  # 设置模型为评估模式
        total_loss = 0
        with torch.no_grad():
            for src, trg in eval_loader:
                output = self(src, trg[:, :-1])  # trg[:-1]是因为我们不需要最后一个token
                loss = criterion(output.view(-1, output.size(-1)), trg[:, 1:].contiguous().view(-1))  # 计算损失
                total_loss += loss.item()
        print(f'Evaluation Loss: {total_loss/len(eval_loader)}')
```

# 10，模型保存和加载
```python
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()  # 设置模型为评估模式
        print(f'Model loaded from {path}')
```





