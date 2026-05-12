# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig

from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector
# TODO: 定义
from janus.models.masked_transformer import (
    MultimodalTokenFusionEncoder,
    TransformerWithSimilarityMask,
    create_multimodal_fusion_encoder
)

from janus.models.dual_modality_fusion_encoder import (
    DualModalityTokenFusionEncoder,
    create_dual_modality_fusion_encoder
)

class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

        # 初始化mask generator实例用于图像和文本embedding预处理
        # 获取embedding维度 - 从language model获取
        embed_dim = self.language_model.config.hidden_size

        # 创建visual mask generator
        visual_base_transformer = create_multimodal_fusion_encoder(
            emb_dim=embed_dim,
            n_head=16,  
            ff_dim=embed_dim * 4,
            n_layer=2,  
            n_token=64,
            dropout=0.1
        )
        
        self.visual_mask_generator = TransformerWithSimilarityMask(
            base_transformer=visual_base_transformer,
            similarity_mask_config={
                'temperature': 1.0,
                'hard': False,
                'similarity_eps': 1e-8
            }
        )
        
        # 创建text mask generator
        text_base_transformer = create_multimodal_fusion_encoder(
            emb_dim=embed_dim,
            n_head=16,  
            ff_dim=embed_dim * 4,
            n_layer=2, 
            n_token=30, 
            dropout=0.1
        )
        
        self.text_mask_generator = TransformerWithSimilarityMask(
            base_transformer=text_base_transformer,
            similarity_mask_config={
                'temperature': 1.0,
                'hard': False,
                'similarity_eps': 1e-8
            }
        )

        print('='*50)
        print(self.text_mask_generator)

        # 创建双模态融合编码器
        self.dual_modality_fusion_encoder = create_dual_modality_fusion_encoder(
            emb_dim=embed_dim,
            n_head=16,
            ff_dim=embed_dim * 4,
            n_layer=4,  
            n_image_token=64,  
            n_text_token=30,   
            dropout=0.1
        )

        print('='*50)
        print(self.dual_modality_fusion_encoder)



    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        titles_seq_mask: torch.BoolTensor = None,  
        titles_emb_mask: torch.BoolTensor = None,
        title_tokens: torch.LongTensor = None,  # 统一的title tokens tensor [batch_size, max_n_titles, num_title_tokens]
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        device = next(self.vision_model.parameters()).device
        input_ids = input_ids.to(device) ##

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images = images.bfloat16().to(device) ###
        # images_embeds = self.aligner(self.vision_model(images))
        all_labels = self.gen_vision_model.encode(images)[-1][-1].view(bs*n, -1)
        images_embeds = self.prepare_gen_img_embeds(all_labels)

        # 使用visual_mask_generator对图像embedding进行预处理
        # [b x n, T2, D] -> [b, n, T2, D]
        images_embeds_4d = rearrange(images_embeds, "(b n) t d -> b n t d", b=bs, n=n)  # [b, n, T2, D]
        # 通过visual_mask_generator处理图像embedding
        images_embeds_processed = self.visual_mask_generator(images_embeds_4d)  # [b, n, T2, D]

        # 处理没有title的情况 - 直接处理图像特征
        if titles_seq_mask is None or not titles_seq_mask.any():
            images_embeds = rearrange(images_embeds_processed, "b n t d -> b (n t) d")  # [b, n x T2, D]
            print(f"仅图像模式 - 图像特征: {images_embeds.shape}")

        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")


        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        input_ids = input_ids.clone() ####
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 处理title embeddings
        if titles_seq_mask is not None and titles_emb_mask is not None and title_tokens is not None:
            
            # 批量处理title tokens，严格按照image embedding的处理方式
            # title_tokens shape: [batch_size, max_n_titles, num_title_tokens]
            title_tokens = title_tokens.to(device)
    
            bs, n_titles = title_tokens.shape[0:2]
            # 重新整理为 [b x n_titles, num_title_tokens] 进行批量embedding
            title_tokens_flat = rearrange(title_tokens, "b n t -> (b n) t")
            title_embeds_flat = self.language_model.get_input_embeddings()(title_tokens_flat)
            # title_embeds_flat shape: [b x n_titles, num_title_tokens, embed_dim]

            # 使用text_mask_generator对title embedding进行预处理
            # [b x n_titles, num_title_tokens, embed_dim] -> [b, n_titles, num_title_tokens, embed_dim]
            title_embeds_4d = rearrange(title_embeds_flat, "(b n) t d -> b n t d", b=bs, n=n_titles)  # [b, n_titles, num_title_tokens, embed_dim]
            title_embeds_processed = self.text_mask_generator(title_embeds_4d)  # [b, n_titles, num_title_tokens, embed_dim]
            
            # 双模态融合
            if hasattr(self, 'dual_modality_fusion_encoder'):
                fused_features = self.dual_modality_fusion_encoder(
                    image_tokens=images_embeds_processed,  # [b, n, T2, D] 
                    text_tokens=title_embeds_processed     # [b, n_titles, num_title_tokens, embed_dim]
                )
                
                B, S_total, D = fused_features.shape
                n_img_tokens = images_embeds_processed.shape[1] * images_embeds_processed.shape[2]  # n * T2
                n_text_tokens = title_embeds_processed.shape[1] * title_embeds_processed.shape[2]   # n_titles * num_title_tokens
                
                enhanced_image_features = fused_features[:, :n_img_tokens, :]    # [B, n*T2, D]
                enhanced_text_features = fused_features[:, n_img_tokens:, :]     # [B, n_titles*num_title_tokens, D]
                
                images_embeds = enhanced_image_features
                title_embeds = enhanced_text_features
                
            else:
                images_embeds = rearrange(images_embeds_processed, "b n t d -> b (n t) d")  # [b, n x T2, D]
                title_embeds = rearrange(title_embeds_processed, "b n t d -> b (n t) d")  # [b, n_titles x num_title_tokens, embed_dim]            

            # titles_emb_mask -> [b, n_titles x num_title_tokens]
            titles_emb_mask_flat = rearrange(titles_emb_mask, "b n t -> b (n t)")
            
            # 替换title placeholder tokens with actual title embeddings
            if titles_seq_mask.any() and titles_emb_mask_flat.any():
                title_replacement = title_embeds[titles_emb_mask_flat].to(inputs_embeds.dtype).to(inputs_embeds.device)
                replacement_embeds = torch.zeros_like(inputs_embeds, device=inputs_embeds.device)
                replacement_embeds[titles_seq_mask] = title_replacement
                mask_expanded = titles_seq_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = torch.where(mask_expanded, replacement_embeds, inputs_embeds)


        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask].to(inputs_embeds.dtype)###

        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))


AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)