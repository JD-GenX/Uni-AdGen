import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union
from enum import Enum

class ModalityType(Enum):
    """模态类型枚举"""
    IMAGE = "image"
    TEXT = "text"

class DualModalityTokenFusionEncoder(nn.Module):
    """
    双模态Token融合Encoder
    
    输入格式:
    - 图像token: [B, N_img, T_img, D] - B个batch，N_img张图像，每张T_img个token，D维特征
    - 文本token: [B, N_txt, T_txt, D] - B个batch，N_txt个文本序列，每个T_txt个token，D维特征
    
    输出格式:
    - 融合后的多模态特征表示
    """
    
    def __init__(
        self,
        emb_dim: int,
        n_head: int, 
        ff_dim: int,
        n_layer: int = 4,
        # 图像相关参数
        n_image_token: int = 196,  # 每张图像的token数量 (如14x14 patches)
        # 文本相关参数  
        n_text_token: int = 77,    # 每个文本序列的token数量
        # 其他参数
        proj: bool = True,
        extra_proj: bool = False,
        encode_ratio: Optional[float] = None,
        activation: str = "quick_gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.ff_dim = ff_dim
        self.n_layer = n_layer
        self.n_image_token = n_image_token
        self.n_text_token = n_text_token
        self.encode_ratio = encode_ratio
        
        # 编码比例处理
        self.pre_proj = self.post_proj = None
        if encode_ratio and encode_ratio != 1:
            self.pre_proj = nn.Linear(emb_dim, int(emb_dim // encode_ratio))
            self.post_proj = nn.Linear(int(emb_dim // encode_ratio), emb_dim)
            emb_dim = int(emb_dim // encode_ratio)
            n_head = int(n_head // encode_ratio)
        
        # 激活函数
        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "quick_gelu":
            self.act = QuickGELU()
        else:
            self.act = activation
        
        # 图像位置编码
        self.image_pos_encoding = nn.Parameter(torch.empty(1, n_image_token, emb_dim))
        
        # 文本位置编码
        self.text_pos_encoding = nn.Parameter(torch.empty(1, n_text_token, emb_dim))
        
        # 模态类型嵌入
        self.modality_embeddings = nn.Embedding(2, emb_dim)  # 0: image, 1: text

        self.proj = None
        if proj:
            self.proj = nn.Sequential(
                nn.Linear(emb_dim, ff_dim),
                self.act,
                nn.Linear(ff_dim, emb_dim),
            )

        self.cross_modal_layers = nn.ModuleList([
            CrossModalTransformerBlock(
                emb_dim, n_head, ff_dim,
                activation=activation,
                dropout=dropout
            ) for _ in range(n_layer)
        ])

        self.extra_proj = None
        if extra_proj:
            self.extra_proj = nn.ModuleList([
                nn.Linear(emb_dim, emb_dim) for _ in range(n_layer)
            ])

        self.final_norm = nn.LayerNorm(emb_dim)
        self.output_proj = nn.Linear(emb_dim, emb_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """参数初始化"""
        nn.init.normal_(self.image_pos_encoding, std=0.01)
        nn.init.normal_(self.text_pos_encoding, std=0.01)

        nn.init.normal_(self.modality_embeddings.weight, std=0.02)

        for proj in [self.pre_proj, self.post_proj]:
            if proj is not None:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.constant_(proj.bias, 0.)
        
        if self.proj is not None:
            nn.init.xavier_uniform_(self.proj[0].weight)
            nn.init.xavier_uniform_(self.proj[2].weight)
            nn.init.constant_(self.proj[0].bias, 0.)
            nn.init.constant_(self.proj[2].bias, 0.)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)
        
        if self.extra_proj is not None:
            for layer in self.extra_proj:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.)
    
    def apply_modality_specific_positional_encoding(
        self, 
        tokens: torch.Tensor, 
        modality: ModalityType,
        batch_size: int,
        num_sequences: int
    ) -> torch.Tensor:
        """
        为指定模态应用独立的位置编码
        
        Args:
            tokens: [B, N, T, D] 输入token序列
            modality: 模态类型 (IMAGE 或 TEXT)
            batch_size: 批次大小
            num_sequences: 序列数量（图像数或文本序列数）
            
        Returns:
            [B, N, T, D] 添加位置编码后的序列
        """
        B, N, T, D = tokens.shape
        
        # 选择对应模态的位置编码
        if modality == ModalityType.IMAGE:
            pos_encoding = self.image_pos_encoding
            expected_tokens = self.n_image_token
            modality_id = 0
        elif modality == ModalityType.TEXT:
            pos_encoding = self.text_pos_encoding
            expected_tokens = self.n_text_token
            modality_id = 1
        else:
            raise ValueError(f"不支持的模态类型: {modality}")
        
        # 处理token长度不匹配的情况
        if T != expected_tokens:
            if T < expected_tokens:
                pos_encoding = pos_encoding[:, :T, :]
            else:
                repeat_times = (T + expected_tokens - 1) // expected_tokens
                pos_encoding = pos_encoding.repeat(1, repeat_times, 1)[:, :T, :]
        
        # [1, T, D] -> [B, N, T, D]
        pos_encoding = pos_encoding.repeat(B, N, 1, 1)
        
        modality_emb = self.modality_embeddings(
            torch.full((B, N, T), modality_id, device=tokens.device, dtype=torch.long)
        )

        return tokens + pos_encoding + modality_emb
    
    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        return_separate_modalities: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            image_tokens: [B, N_img, T_img, D] 图像token序列
            text_tokens: [B, N_txt, T_txt, D] 文本token序列  
            image_mask: 图像注意力掩码
            text_mask: 文本注意力掩码
            return_separate_modalities: 是否返回各模态的独立特征
            
        Returns:
            如果return_separate_modalities=False:
                融合后的特征 [B, S_total, D] (S_total = N_img*T_img + N_txt*T_txt)
            如果return_separate_modalities=True:
                (fused_features, enhanced_image_features, enhanced_text_features)
        """
        if self.encode_ratio is not None:
            image_tokens = self.pre_proj(image_tokens)
            text_tokens = self.pre_proj(text_tokens)
        
        B_img, N_img, T_img, D = image_tokens.shape
        B_txt, N_txt, T_txt, D = text_tokens.shape

        assert B_img == B_txt, f"图像和文本的批次大小不一致: {B_img} vs {B_txt}"
        B = B_img
        
        image_with_pe = self.apply_modality_specific_positional_encoding(
            image_tokens, ModalityType.IMAGE, B, N_img
        )
        
        text_with_pe = self.apply_modality_specific_positional_encoding(
            text_tokens, ModalityType.TEXT, B, N_txt
        )

        if self.proj is not None:
            image_with_pe = self.proj(image_with_pe)
            text_with_pe = self.proj(text_with_pe)
        
        # [B, S, D]
        img_flat = image_with_pe.view(B, N_img * T_img, D)  # [B, N_img*T_img, D]
        txt_flat = text_with_pe.view(B, N_txt * T_txt, D)   # [B, N_txt*T_txt, D]
        
        combined_tokens = torch.cat([img_flat, txt_flat], dim=1)  # [B, N_img*T_img + N_txt*T_txt, D]
        
        # 合并掩码
        combined_mask = None
        if image_mask is not None or text_mask is not None:
            if image_mask is None:
                image_mask = torch.ones(B, N_img * T_img, device=combined_tokens.device)
            else:
                image_mask = image_mask.view(B, N_img * T_img)
            
            if text_mask is None:
                text_mask = torch.ones(B, N_txt * T_txt, device=combined_tokens.device)
            else:
                text_mask = text_mask.view(B, N_txt * T_txt)
            
            combined_mask = torch.cat([image_mask, text_mask], dim=1)
        
        x = combined_tokens
        for i in range(self.n_layer):
            context = x
            if self.extra_proj is not None:
                context = self.extra_proj[i](self.act(context))
            
            x = self.cross_modal_layers[i](
                x, context,
                mask=combined_mask
            )
        
        x = self.final_norm(x)
        fused_features = self.output_proj(x)
        
        if self.encode_ratio is not None:
            fused_features = self.post_proj(fused_features)
        
        if return_separate_modalities:
            img_seq_len = N_img * T_img
            enhanced_image_features = fused_features[:, :img_seq_len, :].view(B, N_img, T_img, D)
            enhanced_text_features = fused_features[:, img_seq_len:, :].view(B, N_txt, T_txt, D)
            
            return fused_features, enhanced_image_features, enhanced_text_features
        else:
            return fused_features


class QuickGELU(nn.Module):
    """快速GELU激活函数"""
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class CrossModalTransformerBlock(nn.Module):
    def __init__(
        self, 
        emb_dim: int, 
        n_head: int, 
        ff_dim: int, 
        activation: str = "quick_gelu", 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.attn = CrossModalMultiheadAttention(
            emb_dim, n_head, dropout=dropout
        )
        
        # 激活函数
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
        """参数初始化"""
        nn.init.xavier_uniform_(self.ff[0].weight)
        nn.init.xavier_uniform_(self.ff[2].weight)
        nn.init.constant_(self.ff[0].bias, 0.)
        nn.init.constant_(self.ff[2].bias, 0.)
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        context = context if context is not None else x

        attn_out = self.attn(x, context, context, mask=mask, weight=weight, alpha=alpha)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x


class CrossModalMultiheadAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.div = torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32))
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """参数初始化"""
        for layer in [self.query, self.key, self.value, self.proj]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        b, s = q.shape[:2]
        b2, s2 = k.shape[:2]
        
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        q = q.view(-1, s, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(-1, s2, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(-1, s2, self.n_head, self.d_head).transpose(1, 2)
        
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


def create_dual_modality_fusion_encoder(
    emb_dim: int = 768,
    n_head: int = 12,
    ff_dim: int = 3072,
    n_layer: int = 6,
    n_image_token: int = 196,  # 14x14 image patches
    n_text_token: int = 77,    # typical text sequence length
    dropout: float = 0.1
) -> DualModalityTokenFusionEncoder:

    return DualModalityTokenFusionEncoder(
        emb_dim=emb_dim,
        n_head=n_head,
        ff_dim=ff_dim,
        n_layer=n_layer,
        n_image_token=n_image_token,
        n_text_token=n_text_token,
        proj=True,
        extra_proj=False,
        activation="quick_gelu",
        dropout=dropout
    )


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = create_dual_modality_fusion_encoder(
        emb_dim=768,
        n_head=12,
        ff_dim=3072,
        n_layer=6,
        n_image_token=196,  # 14x14 patches
        n_text_token=77,    # text sequence
        dropout=0.1
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试数据
    B = 2  # batch size
    N_img, T_img = 4, 196  
    N_txt, T_txt = 3, 77   
    D = 768  

    image_tokens = torch.randn(B, N_img, T_img, D).to(device)
    text_tokens = torch.randn(B, N_txt, T_txt, D).to(device)

    print(f"图像tokens: {image_tokens.shape}")
    print(f"文本tokens: {text_tokens.shape}")
    
    with torch.no_grad():
        fused_output = model(image_tokens, text_tokens)
        print(f"融合输出形状: {fused_output.shape}")

        fused_output, img_features, txt_features = model(
            image_tokens, text_tokens, return_separate_modalities=True
        )
        print(f"融合输出形状: {fused_output.shape}")
        print(f"增强图像特征: {img_features.shape}")
        print(f"增强文本特征: {txt_features.shape}")
    
    print(f"图像位置编码形状: {model.image_pos_encoding.shape}")
    print(f"文本位置编码形状: {model.text_pos_encoding.shape}")
    print(f"模态嵌入权重形状: {model.modality_embeddings.weight.shape}")

    model.train()
    image_tokens_grad = torch.randn(B, N_img, T_img, D, requires_grad=True).to(device)
    text_tokens_grad = torch.randn(B, N_txt, T_txt, D, requires_grad=True).to(device)
    image_tokens_grad.retain_grad()
    text_tokens_grad.retain_grad()
    output = model(image_tokens_grad, text_tokens_grad)

    loss = output.sum()
    loss.backward()
    
    print(f"图像tokens梯度存在: {image_tokens_grad.grad is not None}")
    print(f"文本tokens梯度存在: {text_tokens_grad.grad is not None}")
    
    if image_tokens_grad.grad is not None:
        print(f"图像tokens梯度范数: {image_tokens_grad.grad.norm().item():.6f}")
    if text_tokens_grad.grad is not None:
        print(f"文本tokens梯度范数: {text_tokens_grad.grad.norm().item():.6f}")

    param_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"有梯度的参数: {param_with_grad}/{total_params}")

    short_img_tokens = torch.randn(B, 2, 100, D).to(device)
    short_txt_tokens = torch.randn(B, 1, 50, D).to(device)
    
    with torch.no_grad():
        short_output = model(short_img_tokens, short_txt_tokens)
        print(f"短序列融合输出: {short_output.shape}")

    long_img_tokens = torch.randn(B, 3, 300, D).to(device)
    long_txt_tokens = torch.randn(B, 2, 100, D).to(device) 
    
    with torch.no_grad():
        long_output = model(long_img_tokens, long_txt_tokens)
        print(f"长序列融合输出: {long_output.shape}")