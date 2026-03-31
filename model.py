"""
实现完整的 Transformer 模型（编码器-解码器架构）
包含：多头注意力、位置编码、残差连接、层归一化、前馈网络等

组件                文件中的类                      功能
位置编码​           PositionalEncoding              添加序列位置信息
缩放点积注意力​     ScaledDotProductAttention       计算注意力权重
多头注意力​         MultiHeadAttention              并行多个注意力头
前馈网络​           PositionwiseFeedForward         位置独立的前馈网络
编码器层​           EncoderLayer                    自注意力 + 前馈网络
解码器层​           DecoderLayer                    自注意力 + 交叉注意力 + 前馈网络
编码器​             TransformerEncoder              多个编码器层堆叠
解码器​             TransformerDecoder              多个解码器层堆叠
完整模型​           Transformer                     编码器-解码器架构

残差连接 + 层归一化，在每个子层中都有

自注意力：编码器内部，解码器内部
交叉注意力：解码器关注编码器输出
掩码注意力：防止看到未来信息

编码器:
输入 → 嵌入 → 位置编码 → [自注意力 → 前馈网络]×N → 输出
解码器:
输入 → 嵌入 → 位置编码 → [自注意力 → 交叉注意力 → 前馈网络]×N → 线性 → 输出

类型            Q来源           K,V来源            目的                                位置
自注意力​       同一序列         同一序列         理解内部关系                    编码器、解码器自注意力子层
交叉注意力​       解码器          编码器       对齐源语言和目标语言                  解码器交叉注意力子层
缩放点积注意力   不适用          不适用    一种计算注意力权重的数学公式​             被上述两者调用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """位置编码：为序列中的每个位置添加位置信息"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 频率项：10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # 正弦编码（偶数位置）
        pe[:, 0::2] = torch.sin(position * div_term)
        # 余弦编码（奇数位置）
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 batch 维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model] 带位置编码
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, n_heads, seq_len_q, d_k]
            key: [batch_size, n_heads, seq_len_k, d_k]
            value: [batch_size, n_heads, seq_len_v, d_v]
            mask: [batch_size, 1, seq_len_q, seq_len_k] 或 None
        Returns:
            注意力输出和注意力权重
        """
        d_k = query.size(-1)
        
        # 1. 计算 Q·K^T
        scores = torch.matmul(query, key.transpose(-2, -1))  # [batch, heads, q_len, k_len]
        
        # 2. 缩放
        scores = scores / math.sqrt(d_k)
        
        # 3. 应用 mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax 得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [batch, heads, q_len, k_len]
        
        # 5. Dropout
        attn_weights = self.dropout(attn_weights)
        
        # 6. 加权求和
        output = torch.matmul(attn_weights, value)  # [batch, heads, q_len, d_v]
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # 四个线性变换层：W_q, W_k, W_v, W_o
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将输入拆分成多头
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, n_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()
        
        # 重塑: [batch, seq_len, heads, d_k]
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # 转置: [batch, heads, seq_len, d_k]
        x = x.transpose(1, 2)
        
        return x
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """合并多头输出
        Args:
            x: [batch_size, n_heads, seq_len, d_k]
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.size()
        
        # 转置: [batch, seq_len, heads, d_k]
        x = x.transpose(1, 2)
        
        # 重塑: [batch, seq_len, d_model]
        x = x.contiguous().view(batch_size, seq_len, self.d_model)
        
        return x
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len_q, seq_len_k] 或 None
        Returns:
            输出和注意力权重
        """
        batch_size = query.size(0)
        
        # 保存残差连接的输入
        residual = query
        
        # 1. 线性投影
        Q = self.W_q(query)  # [batch, q_len, d_model]
        K = self.W_k(key)    # [batch, k_len, d_model]
        V = self.W_v(value)  # [batch, v_len, d_model]
        
        # 2. 拆分成多头
        Q = self.split_heads(Q)  # [batch, heads, q_len, d_k]
        K = self.split_heads(K)  # [batch, heads, k_len, d_k]
        V = self.split_heads(V)  # [batch, heads, v_len, d_v]
        
        # 3. 缩放点积注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # 4. 合并多头
        attn_output = self.combine_heads(attn_output)  # [batch, q_len, d_model]
        
        # 5. 输出投影
        output = self.W_o(attn_output)  # [batch, q_len, d_model]
        
        # 6. Dropout
        output = self.dropout(output)
        
        # 7. 残差连接 + 层归一化
        output = self.layer_norm(residual + output)
        
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    """
    位置全连接前馈网络（每个位置独立的前馈网络）由两个线性变换组成，中间有一个 ReLU 激活，FFN(x) = max(0, xW1 + b1)W2 + b2
    输入层 → 隐藏层 → 输出层
      ↓        ↓       ↓
    d_model → d_ff → d_model
      512   → 2048  → 512
   """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 扩展维度，PyTorch 的 nn.Linear包含了权重矩阵 + 偏置，这实际上代表了一层变换
        self.fc2 = nn.Linear(d_ff, d_model)  # 恢复维度
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        residual = x
        
        # 1. 第一层线性 + ReLU
        output = self.fc1(x)
        output = F.relu(output)
        output = self.dropout(output)
        
        # 2. 第二层线性
        output = self.fc2(output)
        output = self.dropout(output)
        
        # 3. 残差连接 + 层归一化
        output = self.layer_norm(residual + output)
        
        return output

class EncoderLayer(nn.Module):
    """Transformer 编码器层"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len] 或 None
        Returns:
            输出和自注意力权重
        """
        # 1. 自注意力子层
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        
        # 2. 前馈网络子层
        output = self.feed_forward(attn_output)
        
        return output, attn_weights

class DecoderLayer(nn.Module):
    """Transformer 解码器层"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 解码器输入 [batch_size, tgt_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            src_mask: 源语言mask [batch_size, 1, 1, src_len]
            tgt_mask: 目标语言mask [batch_size, 1, tgt_len, tgt_len]
        Returns:
            输出、自注意力权重、交叉注意力权重
        """
        # 1. 自注意力子层（目标语言）
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        
        # 2. 交叉注意力子层（源语言到目标语言）
        cross_attn_output, cross_attn_weights = self.cross_attn(
            self_attn_output, encoder_output, encoder_output, src_mask
        )
        
        # 3. 前馈网络子层
        output = self.feed_forward(cross_attn_output)
        
        return output, self_attn_weights, cross_attn_weights

class TransformerEncoder(nn.Module):
    """Transformer 编码器（多个编码器层堆叠）"""
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 num_layers: int,
                 vocab_size: int,
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
    
    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        创建padding mask
        Args:
            seq: [batch_size, seq_len]
        Returns:
            mask: [batch_size, 1, 1, seq_len]
        """
        mask = (seq != 0).unsqueeze(1).unsqueeze(2)  # 0 是 padding token
        return mask
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        Args:
            src: 源语言token IDs [batch_size, src_len]
            src_mask: 源语言mask [batch_size, 1, 1, src_len] 或 None
        Returns:
            编码器输出和每层的注意力权重
        """
        # 如果没有提供mask，创建padding mask
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        
        batch_size, src_len = src.size()
        
        # 1. 词嵌入
        token_embeddings = self.token_embedding(src)  # [batch, src_len, d_model]
        
        # 2. 位置编码
        embeddings = self.positional_encoding(token_embeddings)
        
        # 3. Dropout
        x = self.dropout(embeddings)
        
        # 保存各层的注意力权重
        all_attention_weights = []
        
        # 4. 通过多个编码器层
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            all_attention_weights.append(attn_weights)
        
        # 5. 最终层归一化
        output = self.layer_norm(x)
        
        return output, all_attention_weights

class TransformerDecoder(nn.Module):
    """Transformer 解码器（多个解码器层堆叠）"""
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 num_layers: int,
                 vocab_size: int,
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 输出线性层（映射到词汇表）
        self.output_linear = nn.Linear(d_model, vocab_size)
    
    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """创建padding mask"""
        mask = (seq != 0).unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_look_ahead_mask(self, seq_len: int) -> torch.Tensor:
        """
        创建look-ahead mask（防止看到未来的token）
        Args:
            seq_len: 序列长度
        Returns:
            mask: [seq_len, seq_len]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask == 1  # 上三角为True（要mask的位置）
    
    def forward(self,
                tgt: torch.Tensor,
                encoder_output: torch.Tensor,
                encoder_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list, list]:
        """
        Args:
            tgt: 目标语言token IDs [batch_size, tgt_len]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            encoder_mask: 编码器mask [batch_size, 1, 1, src_len]
            tgt_mask: 目标语言mask [batch_size, 1, tgt_len, tgt_len]
        Returns:
            解码器输出、自注意力权重、交叉注意力权重
        """
        batch_size, tgt_len = tgt.size()
        
        # 创建mask（如果没有提供）
        if encoder_mask is None:
            # 使用与编码器相同的padding逻辑
            encoder_mask = torch.ones(batch_size, 1, 1, encoder_output.size(1), device=tgt.device)
        
        if tgt_mask is None:
            # padding mask
            padding_mask = self.create_padding_mask(tgt)
            
            # look-ahead mask
            look_ahead_mask = self.create_look_ahead_mask(tgt_len)
            look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, tgt_len, tgt_len]
            look_ahead_mask = look_ahead_mask.to(tgt.device)
            
            # 合并两个mask
            tgt_mask = padding_mask & look_ahead_mask
        
        # 1. 词嵌入
        token_embeddings = self.token_embedding(tgt)  # [batch, tgt_len, d_model]
        
        # 2. 位置编码
        embeddings = self.positional_encoding(token_embeddings)
        
        # 3. Dropout
        x = self.dropout(embeddings)
        
        # 保存各层的注意力权重
        all_self_attention_weights = []
        all_cross_attention_weights = []
        
        # 4. 通过多个解码器层
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, encoder_output, encoder_mask, tgt_mask
            )
            all_self_attention_weights.append(self_attn_weights)
            all_cross_attention_weights.append(cross_attn_weights)
        
        # 5. 最终层归一化
        output = self.layer_norm(x)
        
        # 6. 线性投影到词汇表
        logits = self.output_linear(output)  # [batch, tgt_len, vocab_size]
        
        return logits, all_self_attention_weights, all_cross_attention_weights

class Transformer(nn.Module):
    """完整的 Transformer 模型（编码器-解码器）"""
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        print(f"初始化 Transformer 模型:")
        print(f"  源语言词表大小: {src_vocab_size}")
        print(f"  目标语言词表大小: {tgt_vocab_size}")
        print(f"  模型维度 (d_model): {d_model}")
        print(f"  注意力头数: {n_heads}")
        print(f"  前馈网络维度: {d_ff}")
        print(f"  编码器层数: {num_encoder_layers}")
        print(f"  解码器层数: {num_decoder_layers}")
        
        # 编码器
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            vocab_size=src_vocab_size,
            max_len=max_len,
            dropout=dropout
        )
        
        # 解码器
        self.decoder = TransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            vocab_size=tgt_vocab_size,
            max_len=max_len,
            dropout=dropout
        )
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建编码器和解码器的mask
        Args:
            src: 源语言token IDs [batch_size, src_len]
            tgt: 目标语言token IDs [batch_size, tgt_len]
        Returns:
            编码器mask和解码器mask
        """
        # 编码器mask（padding mask）
        encoder_mask = self.encoder.create_padding_mask(src)  # [batch, 1, 1, src_len]
        
        # 解码器mask（padding mask + look-ahead mask）
        decoder_padding_mask = self.decoder.create_padding_mask(tgt)  # [batch, 1, 1, tgt_len]
        
        tgt_len = tgt.size(1)
        look_ahead_mask = self.decoder.create_look_ahead_mask(tgt_len)  # [tgt_len, tgt_len]
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, tgt_len, tgt_len]
        look_ahead_mask = look_ahead_mask.to(src.device)
        
        decoder_mask = decoder_padding_mask & look_ahead_mask
        
        return encoder_mask, decoder_mask
    
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        Args:
            src: 源语言token IDs [batch_size, src_len]
            tgt: 目标语言token IDs [batch_size, tgt_len]
        Returns:
            预测的logits和注意力权重
        """
        # 创建mask
        encoder_mask, decoder_mask = self.create_masks(src, tgt)
        
        # 编码
        encoder_output, encoder_attention_weights = self.encoder(src, encoder_mask)
        
        # 解码
        logits, decoder_self_attention_weights, decoder_cross_attention_weights = self.decoder(
            tgt, encoder_output, encoder_mask, decoder_mask
        )
        
        # 收集所有注意力权重
        attention_weights = {
            'encoder': encoder_attention_weights,
            'decoder_self': decoder_self_attention_weights,
            'decoder_cross': decoder_cross_attention_weights
        }
        
        return logits, attention_weights
    
    def generate(self,
                 src: torch.Tensor,
                 max_len: int = 50,
                 bos_token_id: int = 2,
                 eos_token_id: int = 3) -> torch.Tensor:
        """
        自回归生成
        Args:
            src: 源语言token IDs [batch_size, src_len]
            max_len: 最大生成长度
            bos_token_id: 起始token ID
            eos_token_id: 结束token ID
        Returns:
            生成的token IDs [batch_size, generated_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # 起始token
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        # 编码
        encoder_mask = self.encoder.create_padding_mask(src)
        encoder_output, _ = self.encoder(src, encoder_mask)
        
        for _ in range(max_len):
            # 解码当前序列
            logits, _, _ = self.decoder(generated, encoder_output, encoder_mask)
            
            # 取最后一个位置的logits
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # 选择概率最高的token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]
            
            # 添加到生成序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否生成了结束符
            if (next_token == eos_token_id).all():
                break
        
        return generated

def visualize_model_structure():
    """可视化模型结构"""
    print("\n" + "="*60)
    print("Transformer 模型结构概览")
    print("="*60)
    
    # 创建一个小的Transformer实例
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=128,  # 用小维度便于可视化
        n_heads=4,
        d_ff=512,
        num_encoder_layers=3,
        num_decoder_layers=3
    )
    
    print("\n模型组件层次:")
    print("-" * 40)
    
    def print_module_structure(module, prefix="", depth=0):
        """递归打印模块结构"""
        indent = "  " * depth
        
        # 打印当前模块
        module_name = module.__class__.__name__
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{indent}{prefix}{module_name} ({num_params:,} 参数)")
        
        # 递归处理子模块
        for name, child in module.named_children():
            if not list(child.children()):  # 叶子节点
                child_name = child.__class__.__name__
                child_params = sum(p.numel() for p in child.parameters())
                print(f"{indent}  {name}: {child_name} ({child_params:,} 参数)")
            else:
                print_module_structure(child, f"{name}.", depth + 1)
    
    print_module_structure(model)
    
    return model

def test_transformer_forward():
    """测试 Transformer 前向传播"""
    print("\n" + "="*60)
    print("测试 Transformer 前向传播")
    print("="*60)
    
    # 创建模型
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=128,
        n_heads=4,
        d_ff=512,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    # 创建假数据
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(0, 10000, (batch_size, src_len))  # 源语言序列
    tgt = torch.randint(0, 10000, (batch_size, tgt_len))  # 目标语言序列
    
    print(f"\n输入形状:")
    print(f"  源语言: {src.shape}")
    print(f"  目标语言: {tgt.shape}")
    
    # 前向传播
    logits, attention_weights = model(src, tgt)
    
    print(f"\n输出形状:")
    print(f"  Logits: {logits.shape}")  # [batch_size, tgt_len, vocab_size]
    
    print(f"\n注意力权重:")
    print(f"  编码器层数: {len(attention_weights['encoder'])}")
    print(f"  解码器自注意力层数: {len(attention_weights['decoder_self'])}")
    print(f"  解码器交叉注意力层数: {len(attention_weights['decoder_cross'])}")
    
    # 检查第一个编码器层的注意力权重形状
    if len(attention_weights['encoder']) > 0:
        attn_shape = attention_weights['encoder'][0].shape
        print(f"  编码器注意力形状: {attn_shape}")
    
    return model, logits, attention_weights

if __name__ == "__main__":
    # 1. 可视化模型结构
    model = visualize_model_structure()
    
    # 2. 测试前向传播
    model, logits, attention_weights = test_transformer_forward()
    
    # 3. 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n" + "="*60)
    print("模型参数统计")
    print("="*60)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"参数比例: {trainable_params/total_params*100:.2f}%")
    
    # 4. 测试生成
    print(f"\n" + "="*60)
    print("测试自回归生成")
    print("="*60)
    
    # 创建源语言输入
    src_test = torch.randint(0, 10000, (1, 5))
    print(f"源语言输入: {src_test[0].tolist()}")
    
    # 生成翻译
    generated = model.generate(src_test, max_len=10)
    print(f"生成结果: {generated[0].tolist()}")
    
    print(f"\n✅ Transformer 模型实现完成！")