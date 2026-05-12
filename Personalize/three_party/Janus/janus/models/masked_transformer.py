import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

class MultimodalTokenFusionEncoder(nn.Module):
    """
    多模态Token融合Encoder
    
    输入: [N, T, d1] - N张图像，每张T个token，特征维度d1
    输出: [N, T, d1] - 保持相同维度的融合特征
    """
    
    def __init__(
        self, 
        emb_dim: int, 
        n_head: int, 
        ff_dim: int, 
        n_layer: int = 4, 
        n_token: int = 77, 
        proj: bool = False, 
        extra_proj: bool = False, 
        encode_ratio: Optional[float] = None, 
        activation: str = "quick_gelu", 
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_token = n_token
        self.emb_dim = emb_dim
        self.encode_ratio = encode_ratio

        self.pre_proj = self.post_proj = None
        if encode_ratio and encode_ratio != 1:
            self.pre_proj = nn.Linear(emb_dim, int(emb_dim // encode_ratio))
            self.post_proj = nn.Linear(int(emb_dim // encode_ratio), emb_dim)
            emb_dim = int(emb_dim // encode_ratio)
            n_head = int(n_head // encode_ratio)

        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "quick_gelu":
            self.act = QuickGELU()
        else:
            self.act = activation

        self.pos_encoding = nn.Parameter(torch.empty(1, n_token, emb_dim))

        self.proj = None
        if proj:
            self.proj = nn.Sequential(
                nn.Linear(emb_dim, ff_dim),
                self.act,
                nn.Linear(ff_dim, emb_dim),
            )

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                emb_dim, n_head, ff_dim, 
                n_token=n_token, 
                activation=activation, 
                dropout=dropout
            ) for _ in range(n_layer)
        ])

        self.extra_proj = None
        if extra_proj:
            self.extra_proj = nn.ModuleList([
                nn.Linear(emb_dim, emb_dim) for _ in range(n_layer)
            ])
            
        self._reset_parameters()

    def _reset_parameters(self):
        """参数初始化"""
        nn.init.normal_(self.pos_encoding, std=0.01)

        for proj in [self.pre_proj, self.post_proj]:
            if proj is not None:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.constant_(proj.bias, 0.)
                
        if self.proj is not None:
            nn.init.xavier_uniform_(self.proj[0].weight)
            nn.init.xavier_uniform_(self.proj[2].weight)
            nn.init.constant_(self.proj[0].bias, 0.)
            nn.init.constant_(self.proj[2].bias, 0.)
            
        if self.extra_proj is not None:
            for layer in self.extra_proj:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.)

    def apply_independent_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        为每张图像应用独立的位置编码
        
        Args:
            x: [B, N, T, d1] 或 [N, T, d1] 输入token序列
            
        Returns:
            [B, N, T, d1] 或 [N, T, d1] 添加位置编码后的序列（保持输入维度）
        """
        original_shape = x.shape
        
        if len(original_shape) == 4:
            # [B, N, T, d1]
            B, N, T, d1 = original_shape
        elif len(original_shape) == 3:
            # [N, T, d1]
            N, T, d1 = original_shape
            B = 1
        else:
            raise ValueError(f"位置编码输入维度错误。期望 [B, N, T, d1] 或 [N, T, d1]，实际得到 {original_shape}")

        if T != self.n_token:
            if T < self.n_token:
                pos_enc = self.pos_encoding[:, :T, :]
            else:
                repeat_times = (T + self.n_token - 1) // self.n_token
                pos_enc = self.pos_encoding.repeat(1, repeat_times, 1)[:, :T, :]
        else:
            pos_enc = self.pos_encoding

        if len(original_shape) == 4:
            # pos_enc: [1, T, d1] -> [B, N, T, d1]
            pos_enc = pos_enc.repeat(B, N, 1, 1)
        else:
            # pos_enc: [1, T, d1] -> [N, T, d1]
            pos_enc = pos_enc.repeat(N, 1, 1)
        
        return x + pos_enc

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        """   
        Args:
            x: [B, N, T, d1] 或 [N, T, d1] 图像token序列
            mask: 可选的注意力掩码
            weight: 可选的权重
            alpha: 可选的混合系数
            
        Returns:
            [B, N, T, d1] 或 [N, T, d1] 融合后的特征序列（保持输入维度）
        """
        dtype = x.dtype
        original_shape = x.shape

        if len(original_shape) == 4:
            # [B, N, T, d1]
            B, N, T, d1 = original_shape
            is_batch_input = True
        elif len(original_shape) == 3:
            # [N, T, d1]
            N, T, d1 = original_shape
            B = 1
            is_batch_input = False
        else:
            raise ValueError(f"输入张量维度错误。期望 [B, N, T, d1] 或 [N, T, d1]，实际得到 {original_shape}")
        
        # 编码比例预处理
        if self.encode_ratio is not None:
            x = self.pre_proj(x)

        if is_batch_input:

            x = self.apply_independent_positional_encoding(x)

            # [B, N*T, d1]
            x = x.view(B, N * T, -1)
        else:
            x = self.apply_independent_positional_encoding(x)

        if self.proj is not None:
            x = self.proj(x)

        if mask is not None and is_batch_input:
            if mask.dim() == 4:  # [B, N, T] -> [B, N*T]
                mask = mask.view(B, N * T)
        if weight is not None and is_batch_input:
            if weight.dim() == 4:  # [B, N, T] -> [B, N*T]
                weight = weight.view(B, N * T)

        for i in range(self.n_layer):
            context = x
            if self.extra_proj is not None:
                context = self.extra_proj[i](self.act(context))

            x = self.transformer_layers[i](
                x, context,
                mask=mask,
                weight=weight,
                alpha=alpha
            )

        if self.encode_ratio is not None:
            x = self.post_proj(x)
        
        # 恢复原始输入维度
        if is_batch_input:
            # [B, N*T, d1] -> [B, N, T, d1]
            x = x.view(B, N, T, d1)
        
        return x

    def get_attention_weights(
        self, 
        x: torch.Tensor, 
        layer_idx: int = -1
    ) -> torch.Tensor:
        """ 
        Args:
            x: [N, T, d1] 输入序列
            layer_idx: 层索引，-1表示最后一层
            
        Returns:
            注意力权重矩阵
        """
        if layer_idx == -1:
            layer_idx = self.n_layer - 1

        x = self.apply_independent_positional_encoding(x)
        
        for i in range(layer_idx + 1):
            if i == layer_idx:
                return self.transformer_layers[i].attn.forward(x, x, x)
            x = self.transformer_layers[i](x, x)
        
        return None


class QuickGELU(nn.Module):
    """快速GELU激活函数"""
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head, n_token=77, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.n_token = n_token
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        
        self.div = torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32))
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        
        nn.init.constant_(self.query.bias, 0.)
        nn.init.constant_(self.key.bias, 0.)
        nn.init.constant_(self.value.bias, 0.)
        nn.init.constant_(self.proj.bias, 0.)
        
    def forward(self, q, k, v, mask=None, weight=None, alpha=None):
        b, s = q.shape[:2]
        b2, s2 = k.shape[:2]
        
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        
        q = q.view(b, s, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b2, s2, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b2, s2, self.n_head, self.d_head).transpose(1, 2)
        
        score = torch.matmul(q, k.transpose(-2, -1)) / self.div
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            if mask.dim() != score.dim():
                mask = mask.unsqueeze(2)
            score = score * mask
            
        w = self.softmax(score)
        w = self.dropout(w)
        
        out = torch.matmul(w, v)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_head, ff_dim, n_token=77, activation="quick_gelu", dropout=0.1):
        super().__init__()
        self.attn = MultiheadAttention(emb_dim, n_head, n_token=n_token, dropout=dropout)
        
        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "quick_gelu":
            self.act = QuickGELU()
        else:
            self.act = activation
            
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            self.act,
            nn.Linear(ff_dim, emb_dim),
        )
        
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.ff[0].weight)
        nn.init.xavier_uniform_(self.ff[2].weight)
        nn.init.constant_(self.ff[0].bias, 0.)
        nn.init.constant_(self.ff[2].bias, 0.)
        
    def forward(self, x, context=None, mask=None, weight=None, alpha=None):
        context = context if context is not None else x
        
        # 自注意力
        attn_out = self.attn(x, context, context, mask=mask, weight=weight, alpha=alpha)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # 前馈网络
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x


class SimilarityMaskOutputLayer(nn.Module):
    """
    基于相似度的掩码输出层
    
    支持批处理输入：
    - 输入维度: [B, S, D] 或 [B, N*T, D]
    - 输出维度: 与输入维度保持一致
    
    Args:
        d_model: 模型的特征维度
        temperature: Gumbel-Softmax的温度参数
        hard: 是否使用hard Gumbel-Softmax
        similarity_eps: 余弦相似度计算的数值稳定性参数
    """
    
    def __init__(
        self,
        d_model: int,
        temperature: float = 1.0,
        hard: bool = False,
        similarity_eps: float = 1e-8,
        token_keep_rate: float = 0.4
    ):
        super().__init__()
        
        self.d_model = d_model
        self.temperature = temperature
        self.hard = hard
        self.similarity_eps = similarity_eps
        self.token_keep_rate = token_keep_rate
        
        self.output_projection = nn.Linear(d_model, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """参数初始化"""
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0.)
    
    def compute_positional_cosine_similarity(
        self,
        output_tokens: torch.Tensor,
        input_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            output_tokens: [B, S, D] 输出token特征
            input_tokens: [B, S, D] 输入token特征
            
        Returns:
            [B, S] 逐位置的余弦相似度
        """
        output_norm = torch.nn.functional.normalize(output_tokens, p=2, dim=-1, eps=self.similarity_eps)
        input_norm = torch.nn.functional.normalize(input_tokens, p=2, dim=-1, eps=self.similarity_eps)

        similarity = torch.sum(output_norm * input_norm, dim=-1)  # [B, S]
        
        return similarity
    
    def generate_mask_from_similarity(
        self,
        similarity_scores: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Args:
            similarity_scores: [B, S] 逐位置的相似度分数
            temperature: 可选的温度参数
            
        Returns:
            binary_mask: [B, S] 二值掩码
        """
        temp = temperature if temperature is not None else self.temperature
        
        # 使用gumbel_softmax得到概率
        keep_prob = torch.nn.functional.gumbel_softmax(similarity_scores, hard=False, tau=temp, dim=-1)

        # 取top-k,保留固定数量的token
        token_num = similarity_scores.shape[-1]
        keep_token_num = int(self.token_keep_rate * token_num)
        _, indices = torch.topk(keep_prob, keep_token_num, dim=1) # [B, k]
        binary_mask = torch.zeros_like(keep_prob) # [B, H*W]
        binary_mask.scatter_(1, indices, 1.0)

        binary_mask = binary_mask.detach() + keep_prob - keep_prob.detach()

        return binary_mask
    
    def apply_mask(
        self,
        output_tokens: torch.Tensor,
        binary_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        
        Args:
            output_tokens: [B, S, D] 
            binary_mask: [B, S] 
            
        Returns:
            [B, S, D]
        """
        mask_expanded = binary_mask.unsqueeze(-1)  # [B, S, 1]

        masked_tokens = output_tokens * mask_expanded
        
        return masked_tokens
    
    def forward(
        self,
        output_tokens: torch.Tensor,
        input_tokens: torch.Tensor,
        temperature: Optional[float] = None,
        return_attention_info: bool = False
    ) -> torch.Tensor:
        """

        Args:
            output_tokens: [B, N, T, D] 或 [B, S, D] Transformer输出的token特征
            input_tokens: [B, N, T, D] 或 [B, S, D] 输入的token特征
            temperature: 可选的Gumbel-Softmax温度参数
            return_attention_info: 是否返回注意力信息
            
        Returns:
            如果return_attention_info=False:
                [B, N, T, D] 或 [B, S, D] 
            如果return_attention_info=True:
                tuple: (masked_output, similarity_scores, binary_mask)
        """
        original_shape = output_tokens.shape

        if len(original_shape) == 4:
            B, N, T, D = original_shape
            output_tokens = output_tokens.view(B, N * T, D)
            input_tokens = input_tokens.view(B, N * T, D)
            is_batch_input = True
        elif len(original_shape) == 3:
            B, S, D = original_shape
            is_batch_input = False
        else:
            raise ValueError(f"SimilarityMaskOutputLayer输入维度错误。期望 [B, N, T, D] 或 [B, S, D]，实际得到 {original_shape}")

        similarity_scores = self.compute_positional_cosine_similarity(output_tokens, input_tokens)

        binary_mask = self.generate_mask_from_similarity(similarity_scores, temperature)

        masked_tokens = self.apply_mask(output_tokens, binary_mask)

        final_output = self.output_projection(masked_tokens)
        final_output = self.layer_norm(final_output + masked_tokens)

        if is_batch_input:
            # [B, N*T, D] -> [B, N, T, D]
            final_output = final_output.view(B, N, T, D)
            if return_attention_info:
                similarity_scores = similarity_scores.view(B, N, T)
                binary_mask = binary_mask.view(B, N, T)
        
        if return_attention_info:
            return final_output, similarity_scores, binary_mask
        else:
            return final_output
    
    def get_mask_statistics(self, binary_mask: torch.Tensor) -> dict:
        """
        Args:
            binary_mask: [B, S]
            
        Returns:
            dict: 包含掩码统计信息的字典
        """
        batch_size, seq_len = binary_mask.shape
        
        # 计算保留token的比例
        keep_ratio = binary_mask.mean().item()
        
        # 计算每个样本的保留token数量
        keep_counts = binary_mask.sum(dim=-1)  # [B]
        
        stats = {
            'keep_ratio': keep_ratio,
            'mask_ratio': 1.0 - keep_ratio,
            'avg_keep_tokens': keep_counts.float().mean().item(),
            'min_keep_tokens': keep_counts.min().item(),
            'max_keep_tokens': keep_counts.max().item(),
            'std_keep_tokens': keep_counts.float().std().item()
        }
        
        return stats


class TransformerWithSimilarityMask(nn.Module):
    
    def __init__(
        self,
        base_transformer: MultimodalTokenFusionEncoder,
        similarity_mask_config: Optional[dict] = None
    ):
        super().__init__()
        
        self.base_transformer = base_transformer

        default_config = {
            'temperature': 1.0,
            'hard': False,
            'similarity_eps': 1e-8
        }
        
        if similarity_mask_config is not None:
            default_config.update(similarity_mask_config)
        
        # 相似度掩码输出层
        self.similarity_mask_layer = SimilarityMaskOutputLayer(
            d_model=base_transformer.emb_dim,
            **default_config
        )
    
    def forward(
        self,
        input_tokens: torch.Tensor,
        temperature: Optional[float] = None,
        return_attention_info: bool = False,
        **transformer_kwargs
    ) -> torch.Tensor:
        """
        
        Args:
            input_tokens: [B, N, T, D] 或 [B, S, D]
            temperature: 可选的Gumbel-Softmax温度参数
            return_attention_info: 是否返回注意力信息
            **transformer_kwargs: Transformer的其他参数
            
        Returns:
            [B, N, T, D] 或 [B, S, D] 最终输出 或 (output, similarity_scores, binary_mask)
        """
        original_input_tokens = input_tokens.clone()

        transformer_output = self.base_transformer(input_tokens, **transformer_kwargs)

        result = self.similarity_mask_layer(
            output_tokens=transformer_output,
            input_tokens=original_input_tokens,
            temperature=temperature,
            return_attention_info=return_attention_info
        )
        
        return result


def create_multimodal_fusion_encoder(
    emb_dim: int = 768,
    n_head: int = 12,
    ff_dim: int = 3072,
    n_layer: int = 6,
    n_token: int = 196,  
    dropout: float = 0.1
) -> MultimodalTokenFusionEncoder:
    """
    创建多模态融合编码器
    
    Args:
        emb_dim: 嵌入维度
        n_head: 注意力头数
        ff_dim: 前馈网络维度
        n_layer: Transformer层数
        n_token: 每张图像的token数量
        dropout: Dropout率
        
    Returns:
        配置好的多模态融合编码器
    """
    return MultimodalTokenFusionEncoder(
        emb_dim=emb_dim,
        n_head=n_head,
        ff_dim=ff_dim,
        n_layer=n_layer,
        n_token=n_token,
        proj=False,
        extra_proj=False,
        activation="quick_gelu",
        dropout=dropout
    )