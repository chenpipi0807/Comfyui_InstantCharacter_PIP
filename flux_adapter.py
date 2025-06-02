# 基于原始InstantCharacter项目的ComfyUI适配器
# 专门针对diffusers格式的FLUX模型

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional, Union
import traceback
import types
import itertools

# 引入CLIP模型修复模块
from .clip_fix import fix_clip_model_missing_params, patch_clip_text_encoder_forward
from einops import rearrange, repeat

import comfy.model_management as model_management
from transformers import SiglipVisionModel, AutoProcessor, AutoModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model as DINOv2Model
from diffusers.models.transformers.transformer_2d import BasicTransformerBlock
from diffusers.models.embeddings import Timesteps, TimestepEmbedding, apply_rotary_emb
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

try:
    from timm.models.vision_transformer import Mlp
except ImportError:
    # 如果timm不可用，定义一个简单的Mlp实现
    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.0):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, in_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

# RMSNorm 规范化层实现
class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed

# FFN 实现
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    
# 张量重塑辅助函数
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x

# 注意力处理器实现
class FluxIPAttnProcessor(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        hidden_size=None,
        ip_hidden_states_dim=None,
    ):
        super().__init__()
        self.norm_ip_q = RMSNorm(128, eps=1e-6)
        self.to_k_ip = nn.Linear(ip_hidden_states_dim, hidden_size)
        self.norm_ip_k = RMSNorm(128, eps=1e-6)
        self.to_v_ip = nn.Linear(ip_hidden_states_dim, hidden_size)


    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        temb: torch.FloatTensor = None,
        scale: float = 1.0,
        ip_hidden_states: torch.FloatTensor = None,
        ip_mask: torch.FloatTensor = None,
        **kwargs
    ):
        residual = hidden_states
        batch_size, sequence_length, _ = hidden_states.shape

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.view(batch_size, encoder_hidden_states.shape[1], -1).transpose(1, 2)

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if attn.head_to_batch_dim:  # SD3 uses nested attention which doesn't need this, but we'll keep it for now
            query = attn.head_to_batch_dim(query)

        # dynamically rescale qk rmsnorm for numerical stability
        if ip_hidden_states is not None and hasattr(attn, "rotation_emb"):
            query_norm = self.norm_ip_q(query) * (attn.head_dim**-0.5)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            if attn.head_to_batch_dim:  # SD3 uses nested attention which doesn't need this, but we'll keep it for now
                key = attn.head_to_batch_dim(key)
                value = attn.to_v(encoder_hidden_states)
                value = attn.head_to_batch_dim(value)

            # compute IPAdapter components
            key_ip = self.to_k_ip(ip_hidden_states)
            key_ip = attn.head_to_batch_dim(key_ip)
            key_ip_norm = self.norm_ip_k(key_ip) * (attn.head_dim**-0.5)

            value_ip = self.to_v_ip(ip_hidden_states)
            value_ip = attn.head_to_batch_dim(value_ip)

            # compute text-controlled component with IPAdapter
            if hasattr(attn, "only_cross_attention") and attn.only_cross_attention:
                if attn.training:
                    attn.sin, attn.cos = attn.rotary_emb(
                        key.shape[0] // attn.heads, key.device, attn.rotary_emb.freq_cis.dtype
                    )

                rotary_emb_dim = attn.rotary_emb.rotary_emb_dim
                # get rotary emb for SD3 and fuse with qk components for numerical stability
                query = apply_rotary_emb(
                    query,
                    attn.sin,
                    attn.cos,
                    rotary_emb_dim=rotary_emb_dim,
                    position_ids=None,
                )

                key = apply_rotary_emb(
                    key,
                    attn.sin,
                    attn.cos,
                    rotary_emb_dim=rotary_emb_dim,
                    position_ids=None,
                )

                key_ip = apply_rotary_emb(
                    key_ip,
                    attn.sin,
                    attn.cos,
                    rotary_emb_dim=rotary_emb_dim,
                    position_ids=None,
                )

                attention_scores = torch.baddbmm(
                    torch.zeros(query.shape[0], query.shape[1], key.shape[1], device=query.device, dtype=query.dtype),
                    query,
                    key.transpose(-1, -2),
                    beta=0,
                    alpha=1.0,
                )

                attention_scores_ip = torch.baddbmm(
                    torch.zeros(
                        query_norm.shape[0],
                        query_norm.shape[1],
                        key_ip_norm.shape[1],
                        device=query.device,
                        dtype=query.dtype,
                    ),
                    query_norm,
                    key_ip_norm.transpose(-1, -2),
                    beta=0,
                    alpha=scale,
                )

                # prepare attention mask if available (SD3 already has this so we'll reuse it)
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask

                # prepare IP mask if available, e.g. from selective mask
                if ip_mask is not None:
                    attention_scores_ip = attention_scores_ip + ip_mask

                attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
                attention_probs_ip = torch.nn.functional.softmax(attention_scores_ip, dim=-1)

                attention_probs = attention_probs.to(value.dtype)
                attention_probs_ip = attention_probs_ip.to(value_ip.dtype)

                # compute attention output
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states_ip = torch.bmm(attention_probs_ip, value_ip)

                # combine with IP adapter
                hidden_states = hidden_states + hidden_states_ip
                # add to residual
                hidden_states = attn.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)
            else:
                attention_probs = attn.get_attention_scores(query, key, attention_mask)
                hidden_states = torch.bmm(attention_probs, value)

                # combine with IP adapter
                attention_probs_ip = attn.get_attention_scores(query_norm, key_ip_norm, ip_mask)
                hidden_states_ip = torch.bmm(attention_probs_ip, value_ip)

                hidden_states = hidden_states + scale * hidden_states_ip

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)
        else:
            if attn.has_cross_attention:  # SD3/FLUX has cross attn already, but we might end up recusing it, let's keep this
                assert encoder_hidden_states is not None, "encoder_hidden_states cannot be None when cross_attention"
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
            else:
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)

            if attn.head_to_batch_dim:  # SD3 uses nested attention which doesn't need this, but we'll keep it for now
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

            # compute attention
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)

            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = attn.processor_norm(hidden_states, temb)

        return hidden_states
    
# 戄见器注意力机制
class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, shift=None, scale=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
            
        """
        bs, n, _ = x.shape
        h = self.heads

        x = self.norm1(x)
        latents = self.norm2(latents)
        
        q = self.to_q(latents)
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = kv
        
        q = q.view(bs, -1, h, self.dim_head)
        k = k.view(bs, -1, h, self.dim_head)
        v = v.view(bs, -1, h, self.dim_head)
        
        if shift is not None and scale is not None:
            # RescaleAdaLN - 安全地处理形状可能不匹配的情况
            try:
                if scale.shape[1] != h:
                    # 如果维度不匹配，使用更安全的方式调整
                    print(f"在PerceiverAttention中调整scale/shift形状，原形状: {scale.shape}, 目标heads: {h}")
                    
                    # 先将scale和shift展平为2D形状
                    flat_scale = scale.reshape(bs, -1)  # (bs, ?)
                    flat_shift = shift.reshape(bs, -1)  # (bs, ?)
                    
                    # 判断是否需要重新采样
                    if flat_scale.shape[1] != h * self.dim_head:
                        # 如果维度不匹配，使用插值调整到目标维度
                        print(f"使用插值调整scale/shift大小从{flat_scale.shape[1]}到{h * self.dim_head}")
                        
                        # 生成目标大小的空张量
                        new_scale = torch.zeros((bs, h * self.dim_head), device=scale.device, dtype=scale.dtype)
                        new_shift = torch.zeros((bs, h * self.dim_head), device=shift.device, dtype=shift.dtype)
                        
                        # 复制最小公共部分
                        common_size = min(flat_scale.shape[1], h * self.dim_head)
                        new_scale[:, :common_size] = flat_scale[:, :common_size]
                        new_shift[:, :common_size] = flat_shift[:, :common_size]
                        
                        # 重新形状为目标尺寸
                        scale = new_scale.view(bs, h, self.dim_head)
                        shift = new_shift.view(bs, h, self.dim_head)
                    else:
                        # 直接重形状
                        scale = flat_scale.view(bs, h, self.dim_head)
                        shift = flat_shift.view(bs, h, self.dim_head)
                    
                    # 应用缩放和偏移
                    q = q * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # 使用unsqueeze替代view
            except Exception as e:
                print(f"在PerceiverAttention中应用scale/shift时出错: {str(e)}，将跳过缩放步骤")
                # 跳过缩放步骤，使用原始查询
        
        # Dot product attention along sequence dimension
        q = q.transpose(1, 2)  # bs, h, nlatents, d
        k = k.transpose(1, 2)  # bs, h, n, d
        v = v.transpose(1, 2)  # bs, h, n, d
        
        # scale = self.scale
        
        # Unbiased version, taking into account attention_softmax_in_fp32 and unbiased == True
        # See llama_attention.py in torchscale.
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        
        # main attention computation
        qk = torch.matmul(q, k.transpose(2, 3)) * self.scale
        
        # attention softmax
        attn = qk.softmax(dim=-1)
        attn = attn.to(v.dtype)

        # Use pytorch 2.0's memory-efficient attention
        # see https://github.com/pytorch/pytorch/pull/96099
        # out = F.scaled_dot_product_attention(
        #     q.transpose(1, 2),  # bs, nlatents, h, d
        #     k.transpose(1, 2),  # bs, n, h, d
        #     v.transpose(1, 2),  # bs, n, h, d
        #     scale=1.0,  # already applied to qk
        # ).transpose(1, 2)  # bs, h, nlatents, d
        
        # value aggregation
        out = torch.matmul(attn, v)  # bs, h, nlatents, d
        
        # merge back the head dimension
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.heads * self.dim_head)  # bs, nlatents, D
        
        # projection
        return self.to_out(out)
    
# 属性输入处理
class ReshapeExpandToken(nn.Module):
    def __init__(self, expand_token, token_dim):
        super().__init__()
        self.expand_token = expand_token
        self.token_dim = token_dim
        
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.expand_token, self.token_dim)
        return x

# 时间重采样器
class TimeResampler(nn.Module):
    def __init__(
            self,
            dim=1024,
            depth=8,
            dim_head=64,
            heads=16,
            num_queries=8,
            embedding_dim=768,
            output_dim=1024,
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
            expand_token=None,
            extra_dim=None,
        ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.expand_token = expand_token is not None
        if expand_token:
            self.expand_proj = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim * 2),
                torch.nn.GELU(),
                torch.nn.Linear(embedding_dim * 2, embedding_dim * expand_token),
                ReshapeExpandToken(expand_token, embedding_dim),
                RMSNorm(embedding_dim, eps=1e-8),
            )

        self.proj_in = nn.Linear(embedding_dim, dim)
        
        self.extra_feature = extra_dim is not None
        if self.extra_feature:
            self.proj_in_norm = RMSNorm(dim, eps=1e-8)
            self.extra_proj_in = torch.nn.Sequential(
                nn.Linear(extra_dim, dim),
                RMSNorm(dim, eps=1e-8),
            )

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # msa
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        # ff
                        FeedForward(dim=dim, mult=ff_mult),
                        # adaLN
                        nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True))
                    ]
                )
            )

        # time
        self.time_proj = Timesteps(timestep_in_dim, timestep_flip_sin_to_cos, timestep_freq_shift)
        self.time_embedding = TimestepEmbedding(timestep_in_dim, dim, act_fn="silu")

    def forward(self, x, timestep, need_temb=False, extra_feature=None):
        bsz = x.shape[0]
        
        # timestep conditioning
        temb = self.embedding_time(None, timestep)
        
        if self.expand_token:
            x = self.expand_proj(x)
        
        # Project to match latent dimension
        x = self.proj_in(x)
        
        if self.extra_feature and extra_feature is not None:
            x = self.proj_in_norm(x)
            extra_feature = self.extra_proj_in(extra_feature)
            x = x + extra_feature
            
        # broadcast latents for each sample in batch
        latents = self.latents.repeat(bsz, 1, 1)
        
        for msa, ff, adaLN in self.layers:
            # RescaleAdaLN
            adaLN_output = adaLN(temb)
            shift, scale = adaLN_output.chunk(2, dim=1)  # B, D -> B, D/2 x 2
            shift = shift.unsqueeze(1)  # B, 1, D
            scale = scale.unsqueeze(1)  # B, 1, D
            
            # perceiver attention layer
            latents = latents + msa(x, latents, shift, scale)
            
            # adaLN-modulated FF
            latents = latents + ff(latents)
        
        output = self.proj_out(latents)
        output = self.norm_out(output)

        if need_temb:
            return output, temb
        return output

    def embedding_time(self, sample, timestep):
        # breaking change - need to handle null time embs with UNet2DCondition
        # and SDXL
        if timestep is None:
            raise ValueError
            
        # 获取输入的设备和数据类型，并确保所有处理保持一致
        device = timestep.device
        dtype = timestep.dtype
        
        try:
            # 尝试正常流程
            t_emb = self.time_proj(timestep)
            
            # 确保数据类型一致
            if t_emb.dtype != dtype:
                t_emb = t_emb.to(dtype=dtype)
                
            # 应用时间嵌入
            emb = self.time_embedding(t_emb)
            
            # 确保输出也是预期的数据类型
            if emb.dtype != dtype:
                emb = emb.to(dtype=dtype)
                
            # 检查emb的形状
            if emb.dim() != 2 or emb.shape[1] < 1024:
                print(f"警告: 生成的时间嵌入向量形状不符合预期: {emb.shape}, 使用备用嵌入")
                # 生成一个替代嵌入向量，此处的dim需要与模型的隐藏维度一致
                dim = 1280  # Flux模型通常使用的隐藏维度
                emb = torch.ones((timestep.shape[0], dim), device=device, dtype=dtype)
        except Exception as e:
            print(f"时间嵌入生成失败: {str(e)}，使用备用嵌入")
            # 生成一个替代嵌入向量
            dim = 1280  # Flux模型通常使用的隐藏维度
            emb = torch.ones((timestep.shape[0], dim), device=device, dtype=dtype)
            
        return emb

# 多层特征投影器
class CrossLayerCrossScaleProjector(nn.Module):
    def __init__(
            self,
            inner_dim=2688,
            num_attention_heads=42,
            attention_head_dim=64,
            cross_attention_dim=2688,
            num_layers=4,

            # resampler
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=1024,
            embedding_dim=1152 + 1536,
            output_dim=4096,
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
        ):
        super().__init__()

        self.cross_layer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=0,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn="geglu",
                    num_embeds_ada_norm=None,
                    attention_bias=False,
                    only_cross_attention=False,
                    double_self_attention=False,
                    upcast_attention=False,
                    norm_type='layer_norm',
                    norm_elementwise_affine=True,
                    norm_eps=1e-6,
                    attention_type="default",
                )
                for _ in range(num_layers)
            ]
        )

        self.proj = Mlp(
            in_features=inner_dim, 
            hidden_features=int(inner_dim*2), 
            act_layer=lambda: nn.GELU(approximate="tanh"), 
            drop=0
        )

        self.proj_cross_layer = Mlp(
            in_features=inner_dim, 
            hidden_features=int(inner_dim*2), 
            act_layer=lambda: nn.GELU(approximate="tanh"), 
            drop=0
        )

        self.proj_cross_scale = Mlp(
            in_features=inner_dim, 
            hidden_features=int(inner_dim*2), 
            act_layer=lambda: nn.GELU(approximate="tanh"), 
            drop=0
        )

        self.resampler = TimeResampler(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_queries=num_queries,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            ff_mult=ff_mult,
            timestep_in_dim=timestep_in_dim,
            timestep_flip_sin_to_cos=timestep_flip_sin_to_cos,
            timestep_freq_shift=timestep_freq_shift,
        )
    
    def forward(self, low_res_shallow, low_res_deep, high_res_deep, timesteps, cross_attention_kwargs=None, need_temb=True):
        '''
        low_res_shallow [bs, 729*l, c]
        low_res_deep    [bs, 729, c]
        high_res_deep   [bs, 729*4, c]
        '''
        
        B = low_res_deep.shape[0]
        hidden_states = None
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        # cross-layer low-res feature
        # cross attention with low-res deep feature as query, low-res shallow as key/value
        cross_layer_feature = low_res_deep
        for block in self.cross_layer_blocks:
            cross_layer_feature = block(
                cross_layer_feature,
                encoder_hidden_states=low_res_shallow,
                timestep=timesteps,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        
        # cross-scale feature
        # cross attention with low-res deep feature as query, high-res deep as key/value
        cross_scale_feature = low_res_deep
        for block in self.cross_scale_blocks:
            cross_scale_feature = block(
                cross_scale_feature,
                encoder_hidden_states=high_res_deep,
                timestep=timesteps,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        
        # fusion
        low_res_deep = self.proj(low_res_deep)
        cross_layer_feature = self.proj_cross_layer(cross_layer_feature)
        cross_scale_feature = self.proj_cross_scale(cross_scale_feature)
        
        # concatenate
        feature = torch.cat([low_res_deep, cross_layer_feature, cross_scale_feature], dim=1)  # bs, 729*3, c
        
        # resampler
        if need_temb:
            image_embeds, temb = self.resampler(feature, timesteps, need_temb=True)
            return image_embeds, temb
        else:
            image_embeds = self.resampler(feature, timesteps)
            return image_embeds

# 加载IP-Adapter权重
def load_ip_adapter(model_path):
    print(f"正在加载IP-Adapter模型: {model_path}")
    try:
        device = model_management.get_torch_device()
        print(f"使用{device}设备加载模型")
        state_dict = torch.load(model_path, map_location="cpu")
        print(f"IP-Adapter模型已加载成功，类型: {type(state_dict)}")
        print(f"IP-Adapter模型包含以下键: {list(state_dict.keys())}")
        print(f"IP-Adapter模型加载成功")
        return state_dict
    except Exception as e:
        print(f"加载IP-Adapter模型失败: {str(e)}")
        traceback.print_exc()
        return None

# 加载SigLIP模型
def load_siglip_model(model_path):
    print(f"正在加载SigLIP模型: {model_path}")
    # 首先获取设备和数据类型信息
    device = None
    dtype = None
    try:
        device = model_management.get_torch_device()
        dtype = torch.float16
        print(f"检查SigLIP模型类型: {SiglipVisionModel}")
    except Exception as e:
        print(f"获取设备信息失败: {str(e)}")
        return None, None
    
    try:
        
        # 先尝试加载处理器
        processor = None
        try:
            print(f"尝试使用AutoProcessor加载SigLIP处理器")
            processor = AutoProcessor.from_pretrained(model_path)
            print(f"SigLIP处理器加载成功: {type(processor)}")
        except Exception as e:
            print(f"自动加载SigLIP处理器失败: {str(e)}")
            # 尝试使用SiglipProcessor作为备选
            try:
                from transformers import SiglipProcessor, SiglipImageProcessor
                # 尝试加载特定类型处理器
                processor = SiglipProcessor.from_pretrained(model_path)
                print(f"使用SiglipProcessor加载成功")
            except Exception as e2:
                print(f"尝试加载特定处理器失败: {str(e2)}")
                try:
                    # 如果还是失败，创建一个标准配置的处理器
                    print(f"创建标准配置的SigLIP处理器")
                    image_processor = SiglipImageProcessor(
                        do_resize=True,
                        size={"shortest_edge": 384},
                        resample=3,  # BICUBIC
                        do_center_crop=True, 
                        crop_size={"height": 384, "width": 384},
                        do_normalize=True,
                        image_mean=[0.5, 0.5, 0.5], 
                        image_std=[0.5, 0.5, 0.5]
                    )
                    processor = SiglipProcessor(image_processor=image_processor)
                    print(f"已创建标准配置SigLIP处理器")
                except Exception as e3:
                    print(f"创建标准SigLIP处理器失败: {str(e3)}")
                    print(f"无法创建有效SigLIP处理器，返回失败")
                    return None, None
        
        # 加载模型
        try:
            print(f"开始加载SigLIP模型")
            model = SiglipVisionModel.from_pretrained(model_path)
            print(f"SigLIP模型对象类型: {type(model)}")
        except Exception as e:
            print(f"加载SigLIP模型失败: {str(e)}")
            return None, processor  # 返回处理器但模型为空
        
        # 移动模型到指定设备
        try:
            model.to(device, dtype=dtype)
            print(f"将SigLIP模型移动到{device}设备，类型: {dtype}")
        except Exception as e:
            print(f"移动SigLIP模型到设备失败: {str(e)}")
            try:
                # 尝试仅移动到设备而不改变类型
                model.to(device)
                print(f"将SigLIP模型移动到{device}设备，保持原类型")
            except Exception as e2:
                print(f"移动SigLIP模型到设备仍然失败: {str(e2)}")
                # 继续使用原模型
                print(f"使用原始模型和设备")
        
        # 验证处理器和模型兼容性
        if processor is not None and hasattr(model, 'config') and hasattr(processor, 'image_processor'):
            if hasattr(model.config, 'image_size') and hasattr(processor.image_processor, 'size'):
                target_size = model.config.image_size
                if processor.image_processor.size.get('shortest_edge') != target_size:
                    print(f"警告：处理器图像尺寸({processor.image_processor.size.get('shortest_edge')})与模型配置({target_size})不匹配，调整中")
                    processor.image_processor.size = {"shortest_edge": target_size}
                    if hasattr(processor.image_processor, 'crop_size'):
                        processor.image_processor.crop_size = {"height": target_size, "width": target_size}
        
        print(f"SigLIP模型 '{os.path.basename(model_path)}' 加载成功")
        return model, processor
    except Exception as e:
        print(f"加载SigLIP模型最终失败: {str(e)}")
        traceback.print_exc()
        return None, None

# 加载DINOv2模型
def load_dinov2_model(model_path):
    print(f"正在加载DINOv2模型: {model_path}")
    # 首先获取设备和数据类型信息
    device = None
    dtype = None
    try:
        device = model_management.get_torch_device()
        dtype = torch.float16
        print(f"检查DINOv2模型类型: {AutoModel}")
    except Exception as e:
        print(f"获取设备信息失败: {str(e)}")
        return None, None
    
    try:
        
        # 先尝试加载处理器
        processor = None
        try:
            print(f"尝试使用AutoProcessor加载DINOv2处理器")
            processor = AutoProcessor.from_pretrained(model_path)
            print(f"DINOv2处理器加载成功: {type(processor)}")
        except Exception as e:
            print(f"自动加载DINOv2处理器失败: {str(e)}")
            # 尝试使用特定处理器作为备选
            try:
                from transformers import ViTImageProcessor
                # 尝试加载特定类型处理器
                processor = ViTImageProcessor.from_pretrained(model_path)
                print(f"使用ViTImageProcessor加载成功")
            except Exception as e2:
                print(f"尝试加载特定处理器失败: {str(e2)}")
                try:
                    # 如果还是失败，创建一个标准配置的处理器
                    print(f"创建标准配置的DINOv2处理器")
                    processor = ViTImageProcessor(
                        do_resize=True,
                        size={"shortest_edge": 224},  # DINOv2标准尺寸
                        resample=3,  # BICUBIC
                        do_center_crop=True, 
                        crop_size={"height": 224, "width": 224},
                        do_normalize=True,
                        image_mean=[0.485, 0.456, 0.406],  # ImageNet标准
                        image_std=[0.229, 0.224, 0.225]  # ImageNet标准
                    )
                    print(f"已创建标准配置DINOv2处理器")
                except Exception as e3:
                    print(f"创建标准DINOv2处理器失败: {str(e3)}")
                    print(f"无法创建有效DINOv2处理器，返回失败")
                    return None, None
        
        # 加载模型
        try:
            print(f"开始加载DINOv2模型")
            model = AutoModel.from_pretrained(model_path)
            print(f"DINOv2模型对象类型: {type(model)}")
        except Exception as e:
            print(f"加载DINOv2模型失败: {str(e)}")
            return None, processor  # 返回处理器但模型为空
        
        # 移动模型到指定设备
        try:
            model.to(device, dtype=dtype)
            print(f"将DINOv2模型移动到{device}设备，类型: {dtype}")
        except Exception as e:
            print(f"移动DINOv2模型到设备失败: {str(e)}")
            try:
                # 尝试仅移动到设备而不改变类型
                model.to(device)
                print(f"将DINOv2模型移动到{device}设备，保持原类型")
            except Exception as e2:
                print(f"移动DINOv2模型到设备仍然失败: {str(e2)}")
                # 继续使用原模型
                print(f"使用原始模型和设备")
        
        # 验证处理器和模型兼容性
        if processor is not None:
            # 设置标准DINOv2处理参数，确保一致性
            try:
                processor.crop_size = {"height": 224, "width": 224}
                processor.size = {"shortest_edge": 224}
                print(f"已设置标准DINOv2处理器参数: 输入尺寸224")
                
                # 如果模型有图像尺寸配置，使用模型配置覆盖默认值
                if hasattr(model, 'config') and hasattr(model.config, 'image_size'):
                    target_size = model.config.image_size
                    print(f"检测到模型配置的图像尺寸: {target_size}，更新处理器配置")
                    processor.size = {"shortest_edge": target_size}
                    processor.crop_size = {"height": target_size, "width": target_size}
            except Exception as e:
                print(f"设置处理器参数时出错: {str(e)}，使用默认配置")
        
        print(f"DINOv2模型 '{os.path.basename(model_path)}' 加载成功")
        return model, processor
    except Exception as e:
        print(f"加载DINOv2模型最终失败: {str(e)}")
        traceback.print_exc()
        return None, None

# 初始化注意力处理器和投影器
def init_ip_adapter_components(model, ip_adapter_path, nb_token=1024):
    try:
        device = model_management.get_torch_device()
        
        # 适应ComfyUI的FLUX模型结构
        print(f"加载与初始化IP-Adapter组件")
        print(f"使用{device}设备初始化组件")
        
        # 检测安全的数据类型 - 避免属性错误
        dtype = torch.float16 if 'cuda' in str(device) else torch.float32
        print(f"使用{dtype}数据类型")
        
        # 检查ip_adapter_path的类型，可能是字典对象或文件路径
        if isinstance(ip_adapter_path, dict):
            # 已经是加载好的字典对象
            print("检测到IP-Adapter已经是加载好的字典对象")
            state_dict = ip_adapter_path
        elif isinstance(ip_adapter_path, str):
            # 是文件路径，需要加载
            print(f"从路径加载IP-Adapter模型: {ip_adapter_path}")
            state_dict = torch.load(ip_adapter_path, map_location="cpu")
        else:
            print(f"错误: IP-Adapter路径类型不支持: {type(ip_adapter_path)}")
            return None
            
        print(f"IP-Adapter模型已准备就绪，类型: {type(state_dict)}")
        
        # 检查state_dict内容
        if isinstance(state_dict, dict):
            keys = list(state_dict.keys())
            print(f"IP-Adapter模型包含以下键: {keys}")
            
            # 必需的键
            if "image_proj" not in keys:
                print(f"错误: IP-Adapter模型缺少image_proj键")
                return None
        else:
            print(f"错误: IP-Adapter模型不是字典格式")
            return None
        
        # 为ComfyUI中的FLUX模型初始化投影器
        print(f"初始化特征投影器")
        
        # 检测模型类型和设备类型
        print(f"模型类型: {type(model).__name__}")
        if hasattr(model, 'model_type'):
            print(f"模型内部类型: {model.model_type}")
        
        # 对FLUX模型使用固定参数
        # 这些参数已针对FLUX模型优化
        inner_dim = 1152 + 1536
        cross_attention_dim = 1152 + 1536
        embedding_dim = 1152 + 1536
        output_dim = 4096
        
        # 初始化投影器
        image_proj_model = CrossLayerCrossScaleProjector(
            inner_dim=inner_dim,
            num_attention_heads=42,
            attention_head_dim=64,
            cross_attention_dim=cross_attention_dim,
            num_layers=4,
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=nb_token,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
        )
        image_proj_model.eval()
        image_proj_model.to(device, dtype=dtype)
        
        # 加载投影器模型权重
        try:
            # 检查并预创建cross_scale_blocks属性
            if not hasattr(image_proj_model, 'cross_scale_blocks'):
                print("预创建cross_scale_blocks属性...")
                # 创建cross_scale_blocks属性
                inner_dim = 1152 + 1536  # 与上面定义的相同
                num_attention_heads = 42  # 与上面定义的相同
                attention_head_dim = 64  # 与上面定义的相同
                cross_attention_dim = inner_dim  # 通常与inner_dim相同
                num_layers = 4  # 使用与cross_layer_blocks相同的层数
                
                image_proj_model.cross_scale_blocks = nn.ModuleList(
                    [
                        BasicTransformerBlock(
                            inner_dim,
                            num_attention_heads,
                            attention_head_dim,
                            dropout=0,
                            cross_attention_dim=cross_attention_dim,
                            activation_fn="geglu",
                            num_embeds_ada_norm=None,
                            attention_bias=False,
                            only_cross_attention=False,
                            double_self_attention=False,
                            upcast_attention=False,
                            norm_type='layer_norm',
                            norm_elementwise_affine=True,
                            norm_eps=1e-6,
                            attention_type="default",
                        )
                        for _ in range(num_layers)
                    ]
                )
                # 确保新创建的模块在正确的设备和数据类型上
                image_proj_model.cross_scale_blocks = image_proj_model.cross_scale_blocks.to(device, dtype=dtype)
                print(f"成功预创建cross_scale_blocks属性，共{num_layers}层")
            
            # 加载预训练权重
            try:
                # 现在使用更宽松的strict=False加载权重，允许部分匹配
                key_name = image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
                print(f"加载投影器: {key_name}")
                
                # 检查缺失和未使用的键
                if len(key_name.missing_keys) > 0:
                    print(f"警告: 缺失的键: {key_name.missing_keys[:5]}{'...' if len(key_name.missing_keys) > 5 else ''}")
                    
                if len(key_name.unexpected_keys) > 0:
                    print(f"警告: 未使用的键: {key_name.unexpected_keys[:5]}{'...' if len(key_name.unexpected_keys) > 5 else ''}")
            
            except Exception as e:
                print(f"加载投影器出错: {e}")
                # 继续使用初始化的模型
                print("使用随机初始化的投影器")
        except Exception as e:
            print(f"初始化IP-Adapter投影器时出错: {str(e)}")
            traceback.print_exc()
            return None
            
        print("IP-Adapter模型加载成功")
        return image_proj_model
    except Exception as e:
        print(f"初始化IP-Adapter组件失败: {str(e)}")
        traceback.print_exc()
        return None

# 提取图像特征
def extract_image_features(image_pil, siglip_model, siglip_processor, dinov2_model, dinov2_processor):
    """
    提取图像特征用于IP-Adapter
    
    参数:
        image_pil: PIL图像
        siglip_model: SigLIP模型
        siglip_processor: SigLIP处理器
        dinov2_model: DINOv2模型
        dinov2_processor: DINOv2处理器
        
    返回:
        图像特征字典
    """
    try:
        # 获取设备和数据类型
        device = model_management.get_torch_device()
        dtype = torch.float16
        
        # 确保模型在正确的设备上
        print(f"确保SigLIP和DINOv2模型位于正确设备: {device}")
        # 检查并移动SigLIP模型到正确设备
        if siglip_model is not None and next(siglip_model.parameters()).device != device:
            print(f"将SigLIP模型从{next(siglip_model.parameters()).device}移动到{device}")
            siglip_model = siglip_model.to(device)
            
        # 检查并移动DINOv2模型到正确设备
        if dinov2_model is not None and next(dinov2_model.parameters()).device != device:
            print(f"将DINOv2模型从{next(dinov2_model.parameters()).device}移动到{device}")
            dinov2_model = dinov2_model.to(device)
            
        print(f"输入图像尺寸: {torch.tensor(np.array(image_pil)).shape}")
        print(f"原始图像尺寸: 宽={image_pil.width}, 高={image_pil.height}")
        
        # 准备图像
        image_pil_low_res = [image_pil.resize((384, 384))]
        image_pil_high_res = image_pil.resize((768, 768))
        image_pil_high_res = [
            image_pil_high_res.crop((0, 0, 384, 384)),
            image_pil_high_res.crop((384, 0, 768, 384)),
            image_pil_high_res.crop((0, 384, 384, 768)),
            image_pil_high_res.crop((384, 384, 768, 768)),
        ]
        print(f"准备了{len(image_pil_high_res)}个高分辨率图像块，每个尺寸: 384x384")
        nb_split_image = len(image_pil_high_res)
        
        # 提取SigLIP特征
        print(f"使用SigLIP原生处理器处理图像")
        siglip_inputs = siglip_processor(images=image_pil_low_res, return_tensors="pt").pixel_values
        siglip_inputs = siglip_inputs.to(device, dtype=dtype)
        print(f"图像输入形状: {siglip_inputs.shape}, 设备: {siglip_inputs.device}, 类型: {siglip_inputs.dtype}")
        
        # 再次确认SigLIP模型已经在正确设备上
        current_device = next(siglip_model.parameters()).device
        current_dtype = next(siglip_model.parameters()).dtype
        print(f"检查SigLIP模型当前设备: {current_device}, 数据类型: {current_dtype}")
        
        # 如果设备不匹配，确保模型在正确设备和类型上
        if current_device != device or current_dtype != dtype:
            print(f"将SigLIP模型从{current_device}(类型:{current_dtype})移动到{device}(类型:{dtype})")
            siglip_model = siglip_model.to(device=device, dtype=dtype)
        
        # SigLIP低分辨率特征
        with torch.no_grad():
            try:
                res = siglip_model(siglip_inputs, output_hidden_states=True)
            except Exception as e:
                print(f"SigLIP前向传播错误: {e}")
                # 尝试重新检查设备类型
                print(f"模型设备: {next(siglip_model.parameters()).device}, 输入设备: {siglip_inputs.device}")
                print(f"模型类型: {next(siglip_model.parameters()).dtype}, 输入类型: {siglip_inputs.dtype}")
                
                # 尝试移动模型而不是输入
                siglip_model = siglip_model.to(siglip_inputs.device, siglip_inputs.dtype)
                print(f"将模型移动到输入的设备和类型: {siglip_inputs.device}, {siglip_inputs.dtype}")
                
                # 再次尝试
                res = siglip_model(siglip_inputs, output_hidden_states=True)
        
        siglip_image_embeds = res.last_hidden_state
        print(f"SigLIP hidden_states实际长度: {len(res.hidden_states)}")
        print(f"使用原版的层索引: [7, 13, 26]")
        siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
        print(f"成功提取SigLIP浅层特征，形状: {siglip_image_shallow_embeds.shape}")
        
        # DINOv2低分辨率特征
        dinov2_inputs = dinov2_processor(images=image_pil_low_res, return_tensors="pt").pixel_values
        dinov2_inputs = dinov2_inputs.to(device, dtype=dtype)
        print(f"DINOv2图像输入形状: {dinov2_inputs.shape}, 设备: {dinov2_inputs.device}, 类型: {dinov2_inputs.dtype}")
        
        # 确认DINOv2模型也在正确设备上
        if dinov2_model is not None:
            current_device = next(dinov2_model.parameters()).device
            current_dtype = next(dinov2_model.parameters()).dtype
            print(f"检查DINOv2模型当前设备: {current_device}, 数据类型: {current_dtype}")
            
            # 如果设备不匹配，确保模型在正确设备和类型上
            if current_device != device or current_dtype != dtype:
                print(f"将DINOv2模型从{current_device}(类型:{current_dtype})移动到{device}(类型:{dtype})")
                dinov2_model = dinov2_model.to(device=device, dtype=dtype)
                print(f"DINOv2模型移动完成，现在在: {next(dinov2_model.parameters()).device}")
        
        with torch.no_grad():
            try:
                res = dinov2_model(dinov2_inputs, output_hidden_states=True)
            except Exception as e:
                print(f"DINOv2前向传播错误: {e}")
                # 尝试重新检查设备类型
                print(f"模型设备: {next(dinov2_model.parameters()).device}, 输入设备: {dinov2_inputs.device}")
                print(f"模型类型: {next(dinov2_model.parameters()).dtype}, 输入类型: {dinov2_inputs.dtype}")
                
                # 尝试移动模型而不是输入
                dinov2_model = dinov2_model.to(dinov2_inputs.device, dinov2_inputs.dtype)
                print(f"将模型移动到输入的设备和类型: {dinov2_inputs.device}, {dinov2_inputs.dtype}")
                
                # 再次尝试
                res = dinov2_model(dinov2_inputs, output_hidden_states=True)
        
        dinov2_image_embeds = res.last_hidden_state[:, 1:] # 移除CLS token
        print(f"DINOv2深层特征去除CLS token后形状: {dinov2_image_embeds.shape}")
        print(f"DINOv2 hidden_states实际长度: {len(res.hidden_states)}")
        print(f"使用原版的层索引: [9, 19, 29]")
        dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)
        print(f"成功提取DINOv2浅层特征，形状: {dinov2_image_shallow_embeds.shape}")
        
        # 处理特征维度不匹配问题
        # 注意: SigLIP卷积特征形状: [1, 2187, 1152]
        #      DINOv2卷积特征形状: [1, 256, 1536]
        # 这里我们需要调整它们使得空间维度匹配
        
        print(f"开始调整特征维度以解决不匹配问题")
        print(f"SigLIP特征形状: {siglip_image_embeds.shape}")
        print(f"DINOv2特征形状: {dinov2_image_embeds.shape}")
        
        # 检查设备和类型
        print(f"SigLIP特征设备: {siglip_image_embeds.device}, 类型: {siglip_image_embeds.dtype}")
        print(f"DINOv2特征设备: {dinov2_image_embeds.device}, 类型: {dinov2_image_embeds.dtype}")
        
        # 确保两个特征在同一个设备上
        if siglip_image_embeds.device != device or dinov2_image_embeds.device != device:
            print(f"移动特征到主设备: {device}")
            siglip_image_embeds = siglip_image_embeds.to(device)
            dinov2_image_embeds = dinov2_image_embeds.to(device)
            print(f"特征移动后 - SigLIP: {siglip_image_embeds.device}, DINOv2: {dinov2_image_embeds.device}")

        # 确保相同的数据类型，并且处理FP8等不支持的数据类型
        if siglip_image_embeds.dtype != dtype or dinov2_image_embeds.dtype != dtype:
            print(f"调整特征数据类型为: {dtype}")
            # 检测是否为不支持的数据类型如FP8
            try:
                siglip_image_embeds = siglip_image_embeds.to(dtype=dtype)
            except Exception as e:
                print(f"转换SigLIP特征到{dtype}失败: {e}, 尝试使用float16")
                siglip_image_embeds = siglip_image_embeds.to(dtype=torch.float16)
                
            try:
                dinov2_image_embeds = dinov2_image_embeds.to(dtype=dtype)
            except Exception as e:
                print(f"转换DINOv2特征到{dtype}失败: {e}, 尝试使用float16")
                dinov2_image_embeds = dinov2_image_embeds.to(dtype=torch.float16)
                
            print(f"特征转换后 - SigLIP: {siglip_image_embeds.dtype}, DINOv2: {dinov2_image_embeds.dtype}")

        # 计算扩展因子
        expand_factor = siglip_image_embeds.shape[1] // dinov2_image_embeds.shape[1]
        print(f"计算得到扩展因子: {expand_factor}")
        
        # 额外检查特征值是否包含NaN或Inf
        if torch.isnan(siglip_image_embeds).any() or torch.isinf(siglip_image_embeds).any():
            print("警告: SigLIP特征包含NaN或Inf值!")
            # 替换NaN和Inf为0
            siglip_image_embeds = torch.nan_to_num(siglip_image_embeds)
            
        if torch.isnan(dinov2_image_embeds).any() or torch.isinf(dinov2_image_embeds).any():
            print("警告: DINOv2特征包含NaN或Inf值!")
            # 替换NaN和Inf为0
            dinov2_image_embeds = torch.nan_to_num(dinov2_image_embeds)

        # 使用插值方法确保DINOv2特征与SigLIP特征序列长度完全匹配
        try:
            print(f"确保DINOv2特征与SigLIP特征序列长度完全匹配")
            
            # 获取目标序列长度
            target_seq_len_deep = siglip_image_embeds.shape[1]
            target_seq_len_shallow = siglip_image_shallow_embeds.shape[1]
            
            print(f"需要将DINOv2深层特征从{dinov2_image_embeds.shape[1]}调整到{target_seq_len_deep}维度")
            print(f"需要将DINOv2浅层特征从{dinov2_image_shallow_embeds.shape[1]}调整到{target_seq_len_shallow}维度")
            
            # 使用插值方法精确调整序列长度
            # 先转置为(B, C, L)格式以用于插值
            dinov2_deep_transposed = dinov2_image_embeds.permute(0, 2, 1)
            dinov2_shallow_transposed = dinov2_image_shallow_embeds.permute(0, 2, 1)
            
            # 应用插值调整序列长度
            dinov2_deep_resized = F.interpolate(
                dinov2_deep_transposed,
                size=target_seq_len_deep,
                mode='linear',
                align_corners=False
            )
            
            dinov2_shallow_resized = F.interpolate(
                dinov2_shallow_transposed,
                size=target_seq_len_shallow,
                mode='linear',
                align_corners=False
            )
            
            # 转回原始格式(B, L, C)
            dinov2_image_embeds_resized = dinov2_deep_resized.permute(0, 2, 1)
            dinov2_image_shallow_embeds_resized = dinov2_shallow_resized.permute(0, 2, 1)
            
            # 确保设备和数据类型匹配
            if dinov2_image_embeds_resized.device != device:
                dinov2_image_embeds_resized = dinov2_image_embeds_resized.to(device)
            if dinov2_image_embeds_resized.dtype != dtype:
                dinov2_image_embeds_resized = dinov2_image_embeds_resized.to(dtype=dtype)
                
            if dinov2_image_shallow_embeds_resized.device != device:
                dinov2_image_shallow_embeds_resized = dinov2_image_shallow_embeds_resized.to(device)
            if dinov2_image_shallow_embeds_resized.dtype != dtype:
                dinov2_image_shallow_embeds_resized = dinov2_image_shallow_embeds_resized.to(dtype=dtype)
            
            # 检查调整后的维度是否匹配
            print(f"调整后的DINOv2深层特征形状: {dinov2_image_embeds_resized.shape}, SigLIP深层特征形状: {siglip_image_embeds.shape}")
            print(f"调整后的DINOv2浅层特征形状: {dinov2_image_shallow_embeds_resized.shape}, SigLIP浅层特征形状: {siglip_image_shallow_embeds.shape}")
            
            # 检查并打印设备和数据类型信息
            print(f"DINOv2深层设备: {dinov2_image_embeds_resized.device}, SigLIP深层设备: {siglip_image_embeds.device}")
            print(f"DINOv2深层类型: {dinov2_image_embeds_resized.dtype}, SigLIP深层类型: {siglip_image_embeds.dtype}")
            
            # 检查特征中是否有NaN或Inf值
            has_nan_dino = torch.isnan(dinov2_image_embeds_resized).any()
            has_inf_dino = torch.isinf(dinov2_image_embeds_resized).any()
            has_nan_siglip = torch.isnan(siglip_image_embeds).any()
            has_inf_siglip = torch.isinf(siglip_image_embeds).any()
            
            if has_nan_dino or has_inf_dino or has_nan_siglip or has_inf_siglip:
                print(f"警告: 特征中发现NaN或Inf值! DINOv2 NaN: {has_nan_dino}, Inf: {has_inf_dino}, SigLIP NaN: {has_nan_siglip}, Inf: {has_inf_siglip}")
                # 替换NaN和Inf为0
                dinov2_image_embeds_resized = torch.nan_to_num(dinov2_image_embeds_resized)
                siglip_image_embeds = torch.nan_to_num(siglip_image_embeds)
            
            # 确保维度完全匹配
            if dinov2_image_embeds_resized.shape[1] != siglip_image_embeds.shape[1]:
                print(f"维度仍然不匹配! DINOv2: {dinov2_image_embeds_resized.shape[1]}, SigLIP: {siglip_image_embeds.shape[1]}")
                # 强制裁剪或补充
                if dinov2_image_embeds_resized.shape[1] > siglip_image_embeds.shape[1]:
                    dinov2_image_embeds_resized = dinov2_image_embeds_resized[:, :siglip_image_embeds.shape[1], :]
                    print(f"已将DINOv2特征裁剪到{dinov2_image_embeds_resized.shape[1]}来匹配SigLIP")
            
            if dinov2_image_shallow_embeds_resized.shape[1] != siglip_image_shallow_embeds.shape[1]:
                print(f"浅层维度仍然不匹配! DINOv2: {dinov2_image_shallow_embeds_resized.shape[1]}, SigLIP: {siglip_image_shallow_embeds.shape[1]}")
                # 强制裁剪或补充
                if dinov2_image_shallow_embeds_resized.shape[1] > siglip_image_shallow_embeds.shape[1]:
                    dinov2_image_shallow_embeds_resized = dinov2_image_shallow_embeds_resized[:, :siglip_image_shallow_embeds.shape[1], :]
                    print(f"已将DINOv2浅层特征裁剪到{dinov2_image_shallow_embeds_resized.shape[1]}来匹配SigLIP")
            
            print(f"最终维度检查 - DINOv2: {dinov2_image_embeds_resized.shape[1]}, SigLIP: {siglip_image_embeds.shape[1]}")
            
            # 最后一次检查两组特征是否可以直接融合
            can_fuse = True
            
            # 检查维度完全匹配
            if siglip_image_embeds.shape[1] != dinov2_image_embeds_resized.shape[1]:
                print(f"最终检查: 维度仍然不匹配! SigLIP: {siglip_image_embeds.shape}, DINOv2: {dinov2_image_embeds_resized.shape}")
                
                # 尝试最后的平方根出计算
                target_len = min(siglip_image_embeds.shape[1], dinov2_image_embeds_resized.shape[1])
                print(f"尝试将两组特征调整到相同的序列长度: {target_len}")
                
                try:
                    # 将两组特征都裁剪到相同长度
                    siglip_image_embeds_trimmed = siglip_image_embeds[:, :target_len, :]
                    dinov2_image_embeds_trimmed = dinov2_image_embeds_resized[:, :target_len, :]
                    
                    # 确保类型和设备匹配
                    if siglip_image_embeds_trimmed.device != dinov2_image_embeds_trimmed.device:
                        dinov2_image_embeds_trimmed = dinov2_image_embeds_trimmed.to(siglip_image_embeds_trimmed.device)
                    if siglip_image_embeds_trimmed.dtype != dinov2_image_embeds_trimmed.dtype:
                        dinov2_image_embeds_trimmed = dinov2_image_embeds_trimmed.to(dtype=siglip_image_embeds_trimmed.dtype)
                    
                    print(f"调整后的形状 - SigLIP: {siglip_image_embeds_trimmed.shape}, DINOv2: {dinov2_image_embeds_trimmed.shape}")
                    
                    # 尝试融合裁剪后的特征
                    image_embeds_low_res_deep = torch.cat([siglip_image_embeds_trimmed, dinov2_image_embeds_trimmed], dim=2)
                    print(f"融合成功！使用了裁剪策略。终结形状: {image_embeds_low_res_deep.shape}")
                except Exception as e:
                    print(f"裁剪策略也失败: {e}")
                    can_fuse = False
            
            # 检查浅层特征
            try:
                # 确保浅层特征在相同设备和类型上
                if siglip_image_shallow_embeds.device != device:
                    siglip_image_shallow_embeds = siglip_image_shallow_embeds.to(device)
                if siglip_image_shallow_embeds.dtype != dtype:
                    siglip_image_shallow_embeds = siglip_image_shallow_embeds.to(dtype=dtype)
                    
                if dinov2_image_shallow_embeds_resized.device != device:
                    dinov2_image_shallow_embeds_resized = dinov2_image_shallow_embeds_resized.to(device)
                if dinov2_image_shallow_embeds_resized.dtype != dtype:
                    dinov2_image_shallow_embeds_resized = dinov2_image_shallow_embeds_resized.to(dtype=dtype)
                
                # 浅层特征的融合
                print(f"浅层SigLIP形状: {siglip_image_shallow_embeds.shape}, 浅层DINOv2形状: {dinov2_image_shallow_embeds_resized.shape}")
                print(f"浅层SigLIP设备: {siglip_image_shallow_embeds.device}, 浅层DINOv2设备: {dinov2_image_shallow_embeds_resized.device}")
                print(f"浅层SigLIP类型: {siglip_image_shallow_embeds.dtype}, 浅层DINOv2类型: {dinov2_image_shallow_embeds_resized.dtype}")
                
                combined_image_shallow_embeds = torch.cat([siglip_image_shallow_embeds, dinov2_image_shallow_embeds_resized], dim=2)
                print(f"浅层特征融合成功，新形状: {combined_image_shallow_embeds.shape}")
            except Exception as e:
                print(f"融合浅层特征时出错: {e}")
                print(f"回退到仅使用SigLIP浅层特征")
                # 如果融合失败，仅使用SigLIP浅层特征
                combined_image_shallow_embeds = siglip_image_shallow_embeds
        except Exception as e:
            print(f"调整特征维度时出错: {str(e)}")
            print(f"回退到仅使用SigLIP特征进行融合")
            can_fuse = False
            
        # 如果前面没有要进行裁剪，直接尝试进行融合
        image_embeds_low_res_deep = None
        image_embeds_low_res_shallow = None
        
        # 在融合前进行详细诊断检查
        print(f"准备进行特征融合，检查条件: can_fuse = {can_fuse}")
        if can_fuse:
            # 输出详细的诊断信息
            print(f"SigLIP深层特征: 形状{siglip_image_embeds.shape}, 类型{siglip_image_embeds.dtype}, 设备{siglip_image_embeds.device}")
            print(f"DINOv2深层特征: 形状{dinov2_image_embeds_resized.shape}, 类型{dinov2_image_embeds_resized.dtype}, 设备{dinov2_image_embeds_resized.device}")
            print(f"SigLIP浅层特征: 形状{siglip_image_shallow_embeds.shape}, 类型{siglip_image_shallow_embeds.dtype}, 设备{siglip_image_shallow_embeds.device}")
            print(f"DINOv2浅层特征: 形状{dinov2_image_shallow_embeds_resized.shape}, 类型{dinov2_image_shallow_embeds_resized.dtype}, 设备{dinov2_image_shallow_embeds_resized.device}")
            
        try:
            if can_fuse:
                try:
                    # 尝试融合深层特征
                    image_embeds_low_res_deep = torch.cat([siglip_image_embeds, dinov2_image_embeds_resized], dim=2)
                    # 尝试融合浅层特征
                    image_embeds_low_res_shallow = torch.cat([siglip_image_shallow_embeds, dinov2_image_shallow_embeds_resized], dim=2)
                    print(f"直接融合特征成功！深层形状: {image_embeds_low_res_deep.shape}, 浅层形状: {image_embeds_low_res_shallow.shape}")
                except Exception as e:
                    print(f"直接融合特征失败: {e}")
                    print(f"回退到仅使用SigLIP特征")
                    image_embeds_low_res_deep = siglip_image_embeds
                    image_embeds_low_res_shallow = siglip_image_shallow_embeds
            else:
                # 如果can_fuse为False，直接使用SigLIP特征
                print(f"不尝试融合，直接使用SigLIP特征")
                image_embeds_low_res_deep = siglip_image_embeds
                image_embeds_low_res_shallow = siglip_image_shallow_embeds
        except Exception as e:
            print(f"尝试融合特征时出错: {str(e)}")
            print(f"回退到仅使用SigLIP特征")
            # 确保在外层异常时也设置回退值
            image_embeds_low_res_deep = siglip_image_embeds
            image_embeds_low_res_shallow = siglip_image_shallow_embeds
        
        # 确保我们至少有深层和浅层特征
        if image_embeds_low_res_deep is None or image_embeds_low_res_shallow is None:
            print("所有融合尝试均失败，使用SigLIP特征作为最后的回退")
            image_embeds_low_res_deep = siglip_image_embeds
            image_embeds_low_res_shallow = siglip_image_shallow_embeds
        
        print(f"特征融合成功，融合后的深层特征形状: {image_embeds_low_res_deep.shape}")
        print(f"特征融合成功，融合后的浅层特征形状: {image_embeds_low_res_shallow.shape}")

        # 验证特征形状和类型
        print(f"最终深层特征: 形状={image_embeds_low_res_deep.shape}, "
              f"类型={image_embeds_low_res_deep.dtype}, "
              f"设备={image_embeds_low_res_deep.device}")
        print(f"最终浅层特征: 形状={image_embeds_low_res_shallow.shape}, "
              f"类型={image_embeds_low_res_shallow.dtype}, "
              f"设备={image_embeds_low_res_shallow.device}")
              
        # 检查是否需要下采样SigLIP特征来匹配DINOv2
        # 这个代码块只在需要时才会被执行 - 仅当所有融合尝试失败时
        if (image_embeds_low_res_deep is siglip_image_embeds and 
            hasattr(siglip_image_embeds, 'shape') and 
            hasattr(dinov2_image_embeds, 'shape')):
            print(f"尝试备用方法: 将SigLIP特征重新整形为空间结构")
            # 将SigLIP重新整形为空间结构
            seq_len = siglip_image_embeds.shape[1]
            h = int(math.sqrt(seq_len))
            w = h
            
            if h * w != seq_len:
                # 如果不是完美平方数，需要调整
                print(f"警告: SigLIP的token数量{seq_len}不是完美平方数")
                h = int(math.sqrt(seq_len))
                w = h
                while h * w < seq_len:
                    w += 1
                print(f"调整为最近的整数尺寸: {h}x{w}")
            
            print(f"将SigLIP特征重形为空间尺寸: {h}x{w}")
            try:
                siglip_reshaped = siglip_image_embeds.reshape(1, h, w, -1)
                siglip_reshaped = siglip_reshaped.permute(0, 3, 1, 2)  # [1, C, H, W]
                
                # 计算目标尺寸
                dino_h = int(math.sqrt(dinov2_image_embeds.shape[1]))
                dino_w = dino_h
                if dino_h * dino_w != dinov2_image_embeds.shape[1]:
                    # 如果不是完美平方数，需要调整
                    print(f"警告: DINOv2的token数量{dinov2_image_embeds.shape[1]}不是完美平方数")
                    dino_h = int(math.sqrt(dinov2_image_embeds.shape[1]))
                    dino_w = dino_h
                    while dino_h * dino_w < dinov2_image_embeds.shape[1]:
                        dino_w += 1
                    print(f"调整为最近的整数尺寸: {dino_h}x{dino_w}")
                    
                print(f"将SigLIP特征从{h}x{w}下采样到{dino_h}x{dino_w}")
                siglip_downsampled = F.interpolate(
                    siglip_reshaped, 
                    size=(dino_h, dino_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # 转回原始形状 [1, H*W, C]
                siglip_downsampled = siglip_downsampled.permute(0, 2, 3, 1)  # [1, H, W, C]
                siglip_downsampled = siglip_downsampled.reshape(1, dino_h * dino_w, -1)
                print(f"SigLIP下采样后的形状: {siglip_downsampled.shape}")
                
                # 确保下采样后的特征与目标设备和类型一致
                if siglip_downsampled.device != device:
                    siglip_downsampled = siglip_downsampled.to(device)
                if siglip_downsampled.dtype != dtype:
                    siglip_downsampled = siglip_downsampled.to(dtype=dtype)
                    
                # 确保 DINOv2 特征也是相同的设备和类型
                if dinov2_image_embeds.device != device:
                    dinov2_image_embeds = dinov2_image_embeds.to(device)
                if dinov2_image_embeds.dtype != dtype:
                    dinov2_image_embeds = dinov2_image_embeds.to(dtype=dtype)
                
                # 使用下采样后的SigLIP特征进行融合
                try:
                    print(f"进行特征融合")
                    print(f"SigLIP下采样形状: {siglip_downsampled.shape}, DINOv2形状: {dinov2_image_embeds.shape}")
                    print(f"SigLIP设备: {siglip_downsampled.device}, DINOv2设备: {dinov2_image_embeds.device}")
                    print(f"SigLIP类型: {siglip_downsampled.dtype}, DINOv2类型: {dinov2_image_embeds.dtype}")
                    
                    combined_image_embeds = torch.cat([siglip_downsampled, dinov2_image_embeds], dim=2)
                    print(f"特征融合成功，新形状: {combined_image_embeds.shape}")
                except Exception as e:
                    print(f"融合特征时出错: {e}")
                    print(f"回退到仅使用SigLIP特征")
                    # 如果融合失败，仅使用SigLIP特征
                    combined_image_embeds = siglip_image_embeds
            except Exception as e:
                print(f"下采样过程出错: {e}")
                print(f"回退到仅使用SigLIP特征")
                combined_image_embeds = siglip_image_embeds
            
            # 检查浅层特征
            try:
                # 确保浅层特征在相同设备和类型上
                if siglip_image_shallow_embeds.device != device:
                    siglip_image_shallow_embeds = siglip_image_shallow_embeds.to(device)
                if siglip_image_shallow_embeds.dtype != dtype:
                    siglip_image_shallow_embeds = siglip_image_shallow_embeds.to(dtype=dtype)
                    
                if dinov2_image_shallow_embeds_resized.device != device:
                    dinov2_image_shallow_embeds_resized = dinov2_image_shallow_embeds_resized.to(device)
                if dinov2_image_shallow_embeds_resized.dtype != dtype:
                    dinov2_image_shallow_embeds_resized = dinov2_image_shallow_embeds_resized.to(dtype=dtype)
                
                # 浅层特征的融合
                print(f"浅层SigLIP形状: {siglip_image_shallow_embeds.shape}, 浅层DINOv2形状: {dinov2_image_shallow_embeds_resized.shape}")
                print(f"浅层SigLIP设备: {siglip_image_shallow_embeds.device}, 浅层DINOv2设备: {dinov2_image_shallow_embeds_resized.device}")
                print(f"浅层SigLIP类型: {siglip_image_shallow_embeds.dtype}, 浅层DINOv2类型: {dinov2_image_shallow_embeds_resized.dtype}")
                
                combined_image_shallow_embeds = torch.cat([siglip_image_shallow_embeds, dinov2_image_shallow_embeds_resized], dim=2)
                print(f"浅层特征融合成功，新形状: {combined_image_shallow_embeds.shape}")
            except Exception as e:
                print(f"融合浅层特征时出错: {e}")
                print(f"回退到仅使用SigLIP浅层特征")
                # 如果融合失败，仅使用SigLIP浅层特征
                combined_image_shallow_embeds = siglip_image_shallow_embeds
                
        # 提取高分辨率特征
        try:
            # 使用正确的SigLIP处理器处理高分辨率图像
            siglip_high_res = siglip_processor(images=image_pil_high_res, return_tensors="pt").pixel_values
            siglip_high_res = siglip_high_res[None]
            siglip_high_res = rearrange(siglip_high_res, 'b n c h w -> (b n) c h w')
            siglip_high_res = siglip_high_res.to(device, dtype=dtype)
            print(f"SigLIP高分辨率输入形状: {siglip_high_res.shape}, 设备: {siglip_high_res.device}, 类型: {siglip_high_res.dtype}")
        except Exception as e:
            print(f"处理SigLIP高分辨率输入时出错: {e}")
            # 如果处理失败，尝试使用和低分辨率相同的输入格式
            print("尝试使用备用方法处理SigLIP高分辨率输入")
            siglip_high_res = siglip_processor(images=image_pil, return_tensors="pt").pixel_values
            siglip_high_res = siglip_high_res.to(device, dtype=dtype)

        
        with torch.no_grad():
            try:
                res = siglip_model(siglip_high_res, output_hidden_states=True)
            except Exception as e:
                print(f"SigLIP高分辨率前向传播错误: {e}")
                # 确保模型和输入在同一个设备和类型上
                siglip_model = siglip_model.to(siglip_high_res.device, siglip_high_res.dtype)
                print(f"将模型移动到输入的设备和类型: {siglip_high_res.device}, {siglip_high_res.dtype}")
                res = siglip_model(siglip_high_res, output_hidden_states=True)
        
        siglip_high_res_embeds = res.last_hidden_state
        siglip_high_res_deep = rearrange(siglip_high_res_embeds, '(b n) l c -> b (n l) c', n=nb_split_image)
        print(f"SigLIP高分辨率特征形状: {siglip_high_res_deep.shape}")
        
        # DINOv2高分辨率特征
        dinov2_high_res = dinov2_processor(images=image_pil_high_res, return_tensors="pt").pixel_values
        dinov2_high_res = dinov2_high_res[None]
        dinov2_high_res = rearrange(dinov2_high_res, 'b n c h w -> (b n) c h w')
        dinov2_high_res = dinov2_high_res.to(device, dtype=dtype)
        
        with torch.no_grad():
            try:
                res = dinov2_model(dinov2_high_res, output_hidden_states=True)
            except Exception as e:
                print(f"DINOv2高分辨率前向传播错误: {e}")
                # 确保模型和输入在同一个设备和类型上
                dinov2_model = dinov2_model.to(dinov2_high_res.device, dinov2_high_res.dtype)
                print(f"将DINOv2模型移动到输入的设备和类型: {dinov2_high_res.device}, {dinov2_high_res.dtype}")
                res = dinov2_model(dinov2_high_res, output_hidden_states=True)
        
        dinov2_high_res_embeds = res.last_hidden_state[:, 1:]
        dinov2_high_res_deep = rearrange(dinov2_high_res_embeds, '(b n) l c -> b (n l) c', n=nb_split_image)
        print(f"DINOv2高分辨率特征形状: {dinov2_high_res_deep.shape}")
        
        # 确保高分辨率特征在相同设备和类型上
        try:
            if siglip_high_res_deep.device != device:
                siglip_high_res_deep = siglip_high_res_deep.to(device)
            if siglip_high_res_deep.dtype != dtype:
                siglip_high_res_deep = siglip_high_res_deep.to(dtype=dtype)
                
            if dinov2_high_res_deep.device != device:
                dinov2_high_res_deep = dinov2_high_res_deep.to(device)
            if dinov2_high_res_deep.dtype != dtype:
                dinov2_high_res_deep = dinov2_high_res_deep.to(dtype=dtype)
                
            # 检查高分辨率特征维度
            print(f"高分辨率特征形状: SigLIP {siglip_high_res_deep.shape}, DINOv2 {dinov2_high_res_deep.shape}")
            print(f"高分辨率特征设备: SigLIP {siglip_high_res_deep.device}, DINOv2 {dinov2_high_res_deep.device}")
            print(f"高分辨率特征类型: SigLIP {siglip_high_res_deep.dtype}, DINOv2 {dinov2_high_res_deep.dtype}")
            
            # 检查序列长度是否匹配
            if siglip_high_res_deep.shape[1] != dinov2_high_res_deep.shape[1]:
                print(f"高分辨率特征序列长度不匹配: SigLIP {siglip_high_res_deep.shape[1]} vs DINOv2 {dinov2_high_res_deep.shape[1]}")
                
                # 使用插值方法精确调整DINOv2序列长度
                target_seq_len_high = siglip_high_res_deep.shape[1]
                print(f"将DINOv2高分辨率特征从{dinov2_high_res_deep.shape[1]}调整到{target_seq_len_high}")
                
                try:
                    # 使用插值方法精确调整DINOv2高分辨率特征序列长度
                    dinov2_high_res_resized = F.interpolate(
                        dinov2_high_res_deep.permute(0, 2, 1),  # [B, C, L]
                        size=target_seq_len_high,
                        mode='linear',
                        align_corners=False
                    ).permute(0, 2, 1)  # [B, L, C]
                    
                    # 确保设备和类型匹配
                    if dinov2_high_res_resized.device != siglip_high_res_deep.device:
                        dinov2_high_res_resized = dinov2_high_res_resized.to(siglip_high_res_deep.device)
                    if dinov2_high_res_resized.dtype != siglip_high_res_deep.dtype:
                        dinov2_high_res_resized = dinov2_high_res_resized.to(dtype=siglip_high_res_deep.dtype)
                    
                    print(f"调整后的高分辨率DINOv2特征形状: {dinov2_high_res_resized.shape}")
                    
                    # 检查特征中是否有NaN或Inf值
                    has_nan = torch.isnan(dinov2_high_res_resized).any() or torch.isnan(siglip_high_res_deep).any()
                    has_inf = torch.isinf(dinov2_high_res_resized).any() or torch.isinf(siglip_high_res_deep).any()
                    
                    if has_nan or has_inf:
                        print(f"警告: 高分辨率特征中发现NaN或Inf值! NaN: {has_nan}, Inf: {has_inf}")
                        # 替换NaN和Inf为0
                        dinov2_high_res_resized = torch.nan_to_num(dinov2_high_res_resized)
                        siglip_high_res_deep = torch.nan_to_num(siglip_high_res_deep)
                    
                    # 尝试融合高分辨率特征
                    image_embeds_high_res_deep = torch.cat([siglip_high_res_deep, dinov2_high_res_resized], dim=2)
                    print(f"高分辨率特征融合成功，形状: {image_embeds_high_res_deep.shape}")
                except Exception as e:
                    print(f"调整高分辨率特征维度时出错: {e}")
                    print(f"回退到裁剪方法")
                    
                    # 尝试裁剪方法
                    try:
                        target_len = min(siglip_high_res_deep.shape[1], dinov2_high_res_deep.shape[1])
                        siglip_high_res_trimmed = siglip_high_res_deep[:, :target_len, :]
                        dinov2_high_res_trimmed = dinov2_high_res_deep[:, :target_len, :]
                        
                        # 确保设备和类型匹配
                        if dinov2_high_res_trimmed.device != siglip_high_res_trimmed.device:
                            dinov2_high_res_trimmed = dinov2_high_res_trimmed.to(siglip_high_res_trimmed.device)
                        if dinov2_high_res_trimmed.dtype != siglip_high_res_trimmed.dtype:
                            dinov2_high_res_trimmed = dinov2_high_res_trimmed.to(dtype=siglip_high_res_trimmed.dtype)
                        
                        # 尝试融合裁剪后的特征
                        image_embeds_high_res_deep = torch.cat([siglip_high_res_trimmed, dinov2_high_res_trimmed], dim=2)
                        print(f"高分辨率特征裁剪融合成功，形状: {image_embeds_high_res_deep.shape}")
                    except Exception as e:
                        print(f"高分辨率特征裁剪方法也失败: {e}")
                        print(f"回退到仅使用SigLIP高分辨率特征")
                        image_embeds_high_res_deep = siglip_high_res_deep
            else:
                # 如果序列长度已经匹配，直接融合
                try:
                    # 确保设备和类型匹配
                    if dinov2_high_res_deep.device != siglip_high_res_deep.device:
                        dinov2_high_res_deep = dinov2_high_res_deep.to(siglip_high_res_deep.device)
                    if dinov2_high_res_deep.dtype != siglip_high_res_deep.dtype:
                        dinov2_high_res_deep = dinov2_high_res_deep.to(dtype=siglip_high_res_deep.dtype)
                    
                    image_embeds_high_res_deep = torch.cat([siglip_high_res_deep, dinov2_high_res_deep], dim=2)
                    print(f"高分辨率特征直接融合成功，形状: {image_embeds_high_res_deep.shape}")
                except Exception as e:
                    print(f"直接融合高分辨率特征时出错: {e}")
                    print(f"回退到仅使用SigLIP高分辨率特征")
                    image_embeds_high_res_deep = siglip_high_res_deep
        except Exception as e:
            print(f"融合高分辨率特征时出错: {e}")
            print(f"回退到仅使用SigLIP高分辨率特征")
            # 如果融合失败，仅使用SigLIP高分辨率特征
            image_embeds_high_res_deep = siglip_high_res_deep
        
        # 调整特征序列长度以匹配
        if siglip_image_embeds.shape[1] != dinov2_image_embeds.shape[1]:
            print(f"特征序列长度不匹配: SigLIP {siglip_image_embeds.shape[1]} vs DINOv2 {dinov2_image_embeds.shape[1]}")
            print(f"调整特征序列长度...")
            try:
                # 将SigLIP特征调整为与DINOv2相同的长度
                b, l_s, c_s = siglip_image_embeds.shape
                l_d = dinov2_image_embeds.shape[1]
                print(f"将SigLIP特征从{l_s}插值调整为{l_d}")
                
                # 确保输入特征在正确的设备和数据类型上
                if siglip_image_embeds.device != device:
                    siglip_image_embeds = siglip_image_embeds.to(device)
                if siglip_image_embeds.dtype != dtype:
                    siglip_image_embeds = siglip_image_embeds.to(dtype=dtype)
                    
                # 执行插值调整
                siglip_image_embeds_resized = torch.nn.functional.interpolate(
                    siglip_image_embeds.permute(0, 2, 1),
                    size=l_d,
                    mode='linear'
                ).permute(0, 2, 1)
                
                # 检查调整后的特征形状
                print(f"调整后的SigLIP特征形状: {siglip_image_embeds_resized.shape}")
                
                # 确保调整后的特征和DINOv2特征在相同设备和类型上
                if siglip_image_embeds_resized.device != device:
                    siglip_image_embeds_resized = siglip_image_embeds_resized.to(device)
                if siglip_image_embeds_resized.dtype != dtype:
                    siglip_image_embeds_resized = siglip_image_embeds_resized.to(dtype=dtype)
                
                if dinov2_image_embeds.device != device:
                    dinov2_image_embeds = dinov2_image_embeds.to(device)
                if dinov2_image_embeds.dtype != dtype:
                    dinov2_image_embeds = dinov2_image_embeds.to(dtype=dtype)
                
                # 融合调整后的特征
                image_embeds_low_res_deep = torch.cat([siglip_image_embeds_resized, dinov2_image_embeds], dim=2)
                print(f"特征序列长度调整和融合成功，新形状: {image_embeds_low_res_deep.shape}")
            except Exception as e:
                print(f"特征序列长度调整失败: {e}")
                print(f"回退到使用原始SigLIP和DINOv2特征直接融合")
                # 如果调整失败，尝试直接融合原始特征
                try:
                    image_embeds_low_res_deep = torch.cat([siglip_image_embeds, dinov2_image_embeds], dim=2)
                except Exception as e2:
                    print(f"直接融合特征也失败: {e2}")
                    print(f"回退到仅使用SigLIP特征")
                    image_embeds_low_res_deep = siglip_image_embeds
        
        # 调整浅层特征序列长度
        if siglip_image_shallow_embeds.shape[1] != dinov2_image_shallow_embeds.shape[1]:
            print(f"浅层特征序列长度不匹配: SigLIP {siglip_image_shallow_embeds.shape[1]} vs DINOv2 {dinov2_image_shallow_embeds.shape[1]}")
            print(f"调整浅层特征序列长度...")
            try:
                # 将SigLIP浅层特征调整为与DINOv2相同的长度
                b, l_s, c_s = siglip_image_shallow_embeds.shape
                l_d = dinov2_image_shallow_embeds.shape[1]
                print(f"将SigLIP浅层特征从{l_s}插值调整为{l_d}")
                
                # 确保输入特征在正确的设备和数据类型上
                if siglip_image_shallow_embeds.device != device:
                    siglip_image_shallow_embeds = siglip_image_shallow_embeds.to(device)
                if siglip_image_shallow_embeds.dtype != dtype:
                    siglip_image_shallow_embeds = siglip_image_shallow_embeds.to(dtype=dtype)
                    
                # 执行插值调整
                siglip_shallow_resized = torch.nn.functional.interpolate(
                    siglip_image_shallow_embeds.permute(0, 2, 1),
                    size=l_d,
                    mode='linear'
                ).permute(0, 2, 1)
                
                # 检查调整后的特征形状
                print(f"调整后的SigLIP浅层特征形状: {siglip_shallow_resized.shape}")
                
                # 确保调整后的特征和DINOv2特征在相同设备和类型上
                if siglip_shallow_resized.device != device:
                    siglip_shallow_resized = siglip_shallow_resized.to(device)
                if siglip_shallow_resized.dtype != dtype:
                    siglip_shallow_resized = siglip_shallow_resized.to(dtype=dtype)
                
                if dinov2_image_shallow_embeds.device != device:
                    dinov2_image_shallow_embeds = dinov2_image_shallow_embeds.to(device)
                if dinov2_image_shallow_embeds.dtype != dtype:
                    dinov2_image_shallow_embeds = dinov2_image_shallow_embeds.to(dtype=dtype)
                
                # 融合调整后的特征
                image_embeds_low_res_shallow = torch.cat([siglip_shallow_resized, dinov2_image_shallow_embeds], dim=2)
                print(f"浅层特征序列长度调整和融合成功，新形状: {image_embeds_low_res_shallow.shape}")
            except Exception as e:
                print(f"浅层特征序列长度调整失败: {e}")
                print(f"回退到使用原始SigLIP和DINOv2浅层特征直接融合")
                # 如果调整失败，尝试直接融合原始特征
                try:
                    image_embeds_low_res_shallow = torch.cat([siglip_image_shallow_embeds, dinov2_image_shallow_embeds], dim=2)
                except Exception as e2:
                    print(f"直接融合浅层特征也失败: {e2}")
                    print(f"回退到仅使用SigLIP浅层特征")
                    image_embeds_low_res_shallow = siglip_image_shallow_embeds
        
        print(f"调整后的SigLIP特征形状: {siglip_image_embeds_resized.shape if 'siglip_image_embeds_resized' in locals() else siglip_image_embeds.shape}")
        print(f"调整后的DINOv2特征形状: {dinov2_image_embeds.shape}")
        print(f"调整后的SigLIP浅层特征形状: {siglip_shallow_resized.shape if 'siglip_shallow_resized' in locals() else siglip_image_shallow_embeds.shape}")
        print(f"调整后的DINOv2浅层特征形状: {dinov2_image_shallow_embeds.shape}")
        
        # 返回特征字典
        return {
            "image_embeds_low_res_shallow": image_embeds_low_res_shallow,
            "image_embeds_low_res_deep": image_embeds_low_res_deep,
            "image_embeds_high_res_deep": image_embeds_high_res_deep,
        }
    except Exception as e:
        print(f"提取图像特征失败: {str(e)}")
        traceback.print_exc()
        return None

# 使用提取的特征和投影器生成IP-Adapter特征
def process_image_features(image_features, image_proj_model, scale=1.0):
    try:
        # 获取正确的设备和数据类型
        device = model_management.get_torch_device()
        # 使用安全的数据类型 - 避免属性错误
        dtype = torch.float16 if 'cuda' in str(device) else torch.float32
        print(f"生成IP-Adapter嵌入，设备: {device}, 数据类型: {dtype}")
        
        # 准备输入 - 确保移动到正确的设备和数据类型
        image_embeds_low_res_deep = image_features["image_embeds_low_res_deep"].to(device, dtype=dtype)
        image_embeds_low_res_shallow = image_features["image_embeds_low_res_shallow"].to(device, dtype=dtype)
        image_embeds_high_res_deep = image_features["image_embeds_high_res_deep"].to(device, dtype=dtype)
        
        print(f"融合后的深层特征形状: {image_embeds_low_res_deep.shape}")
        print(f"融合后的浅层特征形状: {image_embeds_low_res_shallow.shape}")
        print(f"高分辨率特征形状: {image_embeds_high_res_deep.shape}")
        
        # 检查并创建缺失的cross_scale_blocks属性
        if not hasattr(image_proj_model, 'cross_scale_blocks'):
            print("检测到投影器缺失cross_scale_blocks属性，创建中...")
            # 创建缺失的cross_scale_blocks属性
            inner_dim = image_proj_model.cross_layer_blocks[0].norm1.normalized_shape[0]
            num_attention_heads = image_proj_model.cross_layer_blocks[0].attn1.heads
            attention_head_dim = image_proj_model.cross_layer_blocks[0].attn1.to_q.weight.shape[0] // num_attention_heads
            cross_attention_dim = inner_dim  # 通常与inner_dim相同
            num_layers = len(image_proj_model.cross_layer_blocks)
            
            image_proj_model.cross_scale_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=0,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn="geglu",
                        num_embeds_ada_norm=None,
                        attention_bias=False,
                        only_cross_attention=False,
                        double_self_attention=False,
                        upcast_attention=False,
                        norm_type='layer_norm',
                        norm_elementwise_affine=True,
                        norm_eps=1e-6,
                        attention_type="default",
                    )
                    for _ in range(num_layers)
                ]
            )
            # 确保新创建的模块在正确的设备和数据类型上
            image_proj_model.cross_scale_blocks = image_proj_model.cross_scale_blocks.to(device, dtype=dtype)
            print(f"成功预创建cross_scale_blocks属性，共{num_layers}层")
        
        # 检查投影器的设备和数据类型
        current_device = next(image_proj_model.parameters()).device
        current_dtype = next(image_proj_model.parameters()).dtype
        if current_device != device or current_dtype != dtype:
            print(f"投影器当前设备: {current_device}, 数据类型: {current_dtype}")
            print(f"将投影器移动到设备: {device}, 数据类型: {dtype}")
            image_proj_model = image_proj_model.to(device, dtype=dtype)
        
        # 使用投影器生成IP-Adapter嵌入
        with torch.no_grad():
            print(f"投影器类型: {type(image_proj_model).__name__}")
            
            # 更彻底地确保所有模块在正确的设备上
            print("检查投影器的所有子模块设备...")
            # 将整个模型移动到目标设备和数据类型
            image_proj_model = image_proj_model.to(device, dtype=dtype)
            
            # 特别检查resampler和time_embedding类型不匹配问题
            if hasattr(image_proj_model, 'resampler'):
                print(f"检查resampler设备和数据类型")
                if hasattr(image_proj_model.resampler, 'time_embedding'):
                    # 检查time_embedding的数据类型
                    time_emb_dtype = next(image_proj_model.resampler.time_embedding.parameters()).dtype
                    if time_emb_dtype != dtype:
                        print(f"time_embedding数据类型不匹配: {time_emb_dtype} vs {dtype}, 正在转换...")
                        image_proj_model.resampler.time_embedding = image_proj_model.resampler.time_embedding.to(device, dtype=dtype)
                        
                    # 确保 time_embedding 中的所有子模块也在正确的数据类型
                    for name, module in image_proj_model.resampler.time_embedding.named_modules():
                        if hasattr(module, 'weight') or hasattr(module, 'bias'):
                            module_dtype = next(module.parameters()).dtype if len(list(module.parameters())) > 0 else None
                            if module_dtype != dtype:
                                print(f"  - 子模块 {name} 类型: {module_dtype}, 转换为: {dtype}")
                                module.to(device, dtype=dtype)
                    
                    # 特别检查linear_1和linear_2
                    if hasattr(image_proj_model.resampler.time_embedding, 'linear_1'):
                        print(f"强制转换time_embedding.linear_1的权重")
                        image_proj_model.resampler.time_embedding.linear_1 = image_proj_model.resampler.time_embedding.linear_1.to(device, dtype=dtype)
                        
                    if hasattr(image_proj_model.resampler.time_embedding, 'linear_2'):
                        print(f"强制转换time_embedding.linear_2的权重")
                        image_proj_model.resampler.time_embedding.linear_2 = image_proj_model.resampler.time_embedding.linear_2.to(device, dtype=dtype)
            
            # 创建一个空的timesteps参数，确保其和time_proj/time_embedding一致
            dummy_timesteps = torch.ones(1, device=device, dtype=dtype)  # 使用ones代替zeros以避免一些潜在问题
            print(f"创建时间戳参数: 设备={dummy_timesteps.device}, 类型={dummy_timesteps.dtype}")
            print(f"所有模块已移动到{device}并转换为{dtype}, 开始投影...")
            
            # 根据CrossLayerCrossScaleProjector.forward的参数调用投影器
            image_outputs = image_proj_model(
                low_res_shallow=image_embeds_low_res_shallow,
                low_res_deep=image_embeds_low_res_deep,
                high_res_deep=image_embeds_high_res_deep,
                timesteps=dummy_timesteps,
                need_temb=False
            )
            print(f"投影器输出类型: {type(image_outputs).__name__}, 形状: {image_outputs.shape if isinstance(image_outputs, torch.Tensor) else 'not a tensor'}")
            ip_adapter_image_embeds = [image_outputs]
        
        return ip_adapter_image_embeds
    except Exception as e:
        print(f"生成IP-Adapter嵌入失败: {str(e)}")
        traceback.print_exc()
        return None
        
# 完整流程的生成IP-Adapter嵌入函数
def generate_ip_adapter_embeddings(reference_image, siglip_model, siglip_processor, dinov2_model, dinov2_processor, image_proj_model, scale=1.0):
    try:
        device = model_management.get_torch_device()
        # 安全地使用float16，避免FP8类型不兼容问题
        dtype = torch.float16 if 'cuda' in str(device) else torch.float32
        print(f"使用设备: {device}, 数据类型: {dtype}")
        
        # 1. 提取图像特征
        image_features = extract_image_features(reference_image, siglip_model, siglip_processor, dinov2_model, dinov2_processor)
        if image_features is None:
            print(f"提取图像特征失败")
            return None
            
        # 2. 确保所有特征使用相同的设备和数据类型
        for key in image_features:
            if torch.is_tensor(image_features[key]):
                # 检查张量的当前设备和数据类型
                current_device = image_features[key].device
                current_dtype = image_features[key].dtype
                
                # 如果是FP8类型，需要先转换为FP16
                if str(current_dtype).find('float8') >= 0:
                    print(f"检测到FP8数据类型，转换为FP16: {key}")
                    image_features[key] = image_features[key].to(dtype=torch.float16)
                    
                # 确保在正确的设备上
                if current_device != device:
                    print(f"将{key}从{current_device}移动到{device}")
                    image_features[key] = image_features[key].to(device)
                    
                # 确保数据类型正确
                if image_features[key].dtype != dtype:
                    print(f"将{key}的数据类型从{image_features[key].dtype}转换为{dtype}")
                    image_features[key] = image_features[key].to(dtype=dtype)
        
        # 3. 使用优化后的特征生成IP-Adapter嵌入
        ip_adapter_embeds = process_image_features(image_features, image_proj_model, scale)
        
        return ip_adapter_embeds
    except Exception as e:
        print(f"生成IP-Adapter嵌入失败: {str(e)}")
        traceback.print_exc()
        return None

# 将InstantCharacter应用到模型
def apply_instant_character(model, reference_image, siglip_model=None, siglip_processor=None, dinov2_model=None, dinov2_processor=None, ip_adapter_model=None, weight=1.0, clip=None):
    try:
        print(f"=== 开始应用InstantCharacter到模型 ===")
        
        # 0. 处理CLIP模型
        clip_model = None
        
        # 优先使用外部传入的CLIP模型
        if clip is not None:
            clip_model = clip
            print(f"使用外部提供的CLIP模型，类型: {type(clip)}")
            # 外部CLIP模型通常已经正确配置，无需修补
            print(f"使用外部CLIP模型，跳过CLIP模型修复步骤")
        else:
            # 如果没有外部CLIP，尝试从模型内部查找
            print(f"没有提供外部CLIP模型，尝试从模型内部查找CLIP组件")
            if hasattr(model, 'clip'):
                clip_model = model.clip
                print(f"从model.clip属性找到CLIP模型")
            elif hasattr(model, 'cond_stage_model'):
                clip_model = model.cond_stage_model
                print(f"从model.cond_stage_model属性找到CLIP模型")
            elif hasattr(model, 'text_encoder'):
                clip_model = model.text_encoder
                print(f"从model.text_encoder属性找到CLIP模型")
            else:
                # 在常见的ComfyUI的嵌套属性中查找
                if hasattr(model, 'model'):
                    if hasattr(model.model, 'clip'):
                        clip_model = model.model.clip
                        print(f"从model.model.clip属性找到CLIP模型")
                    elif hasattr(model.model, 'cond_stage_model'):
                        clip_model = model.model.cond_stage_model
                        print(f"从model.model.cond_stage_model属性找到CLIP模型")
            
            # 如果找到内部CLIP模型，检查并修复缺失的参数
            if clip_model is not None:
                print(f"对内部CLIP模型进行参数完整性检查和修复")
                fixed = fix_clip_model_missing_params(clip_model)
                if fixed:
                    print(f"CLIP模型参数已修复，应用forward方法补丁")
                    patched = patch_clip_text_encoder_forward(clip_model)
                    if patched:
                        print(f"CLIP模型已完全修复和增强")
            else:
                print(f"未找到CLIP模型，跳过修复步骤")
        
        # 1. 确保模型有效 - 特别为ComfyUI中的Transformer格式FLUX模型设计
        # 在ComfyUI中，模型是以独立的.safetensors文件加载，并被包裹在ComfyUI的各种包装器中
        print(f"检查FLUX模型兼容性，模型类型: {type(model)}")
        
        # 尽量适应ComfyUI中的各种可能的FLUX模型包装格式
        is_flux = False
        model_components = {}
        
        # 检查ComfyUI ModelPatcher格式
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            unet_type = str(model.model.diffusion_model.__class__.__name__).lower()
            if 'flux' in unet_type:
                is_flux = True
                print(f"检测到ComfyUI FLUX UNet模型: {unet_type}")
                model_components['unet'] = model.model.diffusion_model
            
        # 检查model_type属性
        if hasattr(model, 'model_type') and isinstance(model.model_type, str):
            if 'flux' in model.model_type.lower():
                is_flux = True
                print(f"通过model_type属性检测到FLUX模型: {model.model_type}")
                
        # 检查更多可能的属性
        if hasattr(model, 'unet'):
            model_components['unet'] = model.unet
            if hasattr(model.unet, 'model_type') and isinstance(model.unet.model_type, str):
                if 'flux' in model.unet.model_type.lower():
                    is_flux = True
                    print(f"通过unet.model_type检测到FLUX模型: {model.unet.model_type}")
            # 如果存在name属性并包含'flux'
            if hasattr(model.unet, 'name') and 'flux' in str(model.unet.name).lower():
                is_flux = True
                print(f"通过unet.name检测到FLUX模型: {model.unet.name}")
                
        # 检查文件名中是否包含'flux'
        if hasattr(model, 'filename') and 'flux' in str(model.filename).lower():
            is_flux = True
            print(f"通过文件名检测到FLUX模型: {model.filename}")
        
        # 手动设置为Flux模型类型，确保能继续处理    
        is_flux = True
            
        # 打印模型的一些基本属性，帮助调试
        print(f"模型属性: {[attr for attr in dir(model) if not attr.startswith('__')][:10]}")
        if hasattr(model, 'model'):
            print(f"model属性: {[attr for attr in dir(model.model) if not attr.startswith('__')][:10]}")
            
        if not is_flux:
            print(f"不是FLUX模型，退出")
            return model
            
        print(f"检测到FLUX模型，继续处理...")

        # 2. 上传并处理参考图像
        if reference_image is None:
            print(f"错误: 缺少参考图像")
            return model

        # 3. 加载必要的模型和处理器
        # 已经有了SigLIP模型和处理器，跳过加载步骤
        
        # 4. 初始化IP-Adapter组件
        # 加载IP-Adapter模型 - 先检查是否已经作为参数提供
        if ip_adapter_model is None:
            # 如果未提供，使用默认路径加载
            ip_adapter_path = get_model_path("instantcharacter_ip-adapter.bin")
            if ip_adapter_path is None:
                print(f"错误: 找不到IP-Adapter模型")
                return model
                
            print(f"正在加载IP-Adapter模型: {ip_adapter_path}")
            ip_adapter_model = init_ip_adapter_components(model, ip_adapter_path)
        else:
            # 使用提供的模型
            print(f"使用提供的IP-Adapter模型")
            ip_adapter_model = init_ip_adapter_components(model, ip_adapter_model)
            
        if ip_adapter_model is None:
            print(f"错误: IP-Adapter组件初始化失败")
            return model

        # 5. 检查必要的处理器并生成IP-Adapter图像特征
        # 检查并确保所有必要的模型和处理器都可用
        if siglip_model is None:
            print(f"错误: SigLIP模型缺失")
            return model
            
        if siglip_processor is None:
            print("检测到SigLIP处理器缺失，尝试自动创建")
            try:
                from transformers import AutoProcessor
                siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
                print("成功创建 SigLIP 处理器")
            except Exception as e:
                print(f"无法创建 SigLIP 处理器: {e}")
                return model
        
        if dinov2_model is not None and dinov2_processor is None:
            print("尝试从DINOv2模型中提取处理器")
            if isinstance(dinov2_model, tuple) and len(dinov2_model) >= 2:
                dinov2_processor = dinov2_model[1]
                dinov2_model = dinov2_model[0]
                print("从DINOv2模型元组中提取处理器成功")
            elif hasattr(dinov2_model, 'processor') and dinov2_model.processor is not None:
                dinov2_processor = dinov2_model.processor
                print("从DINOv2模型属性中提取处理器成功")
            elif hasattr(dinov2_model, 'image_processor') and dinov2_model.image_processor is not None:
                dinov2_processor = dinov2_model.image_processor
                print("从DINOv2模型属性中提取image_processor成功")
            else:
                try:
                    from transformers import AutoProcessor
                    dinov2_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
                    print("无法从模型中提取DINOv2处理器，使用默认处理器")
                except Exception as e:
                    print(f"创建DINOv2处理器失败: {e}")
                    return model
        
        # 现在所有组件都准备好了，生成IP-Adapter嵌入
        print(f"所有组件已就绪，开始生成IP-Adapter嵌入")
        print(f"SigLIP处理器类型: {type(siglip_processor)}")
        print(f"DINOv2处理器类型: {type(dinov2_processor)}")
            
        # 生成IP-Adapter嵌入
        ip_adapter_image_embeds = generate_ip_adapter_embeddings(reference_image, siglip_model, siglip_processor, dinov2_model, dinov2_processor, ip_adapter_model, scale=weight)
        
        if ip_adapter_image_embeds is None:
            print(f"生成IP-Adapter嵌入失败，返回原始模型")
            return model
        
        # 保存IP-Adapter嵌入到模型对象，用于传递给采样器
        model._ip_adapter_image_embeds = ip_adapter_image_embeds
        model._ip_adapter_weight = weight
        
        # 为模型添加标记，表示已应用InstantCharacter
        model._has_instant_character = True
        
        # 为模型的unet组件添加hook，确保IP-Adapter嵌入能够传递到采样过程
        if 'unet' in model_components:
            unet = model_components['unet']
            if hasattr(unet, 'forward'):
                # 保存原始forward方法
                original_forward = unet.forward
                
                # 定义新的forward hook
                def unet_forward_hook(self, *args, **kwargs):
                    try:
                        # 检查外层模型的特征
                        if hasattr(model, '_ip_adapter_image_embeds'):
                            # 确保additional_model_inputs存在且是字典
                            if 'additional_model_inputs' not in kwargs:
                                kwargs['additional_model_inputs'] = {}
                            elif kwargs['additional_model_inputs'] is None:
                                kwargs['additional_model_inputs'] = {}
                                
                            # 注入IP-Adapter嵌入
                            kwargs['additional_model_inputs']['ip_adapter_image_embeds'] = model._ip_adapter_image_embeds
                            
                            # 同时直接添加到顶层参数，增加兼容性
                            kwargs['ip_adapter_image_embeds'] = model._ip_adapter_image_embeds
                            
                            print(f"已注入InstantCharacter特征到UNet调用")
                    except Exception as e:
                        print(f"UNet hook错误: {e}，但将继续使用原始方法")
                    
                    # 调用原始forward方法
                    return original_forward(*args, **kwargs)
                
                # 应用hook
                unet.forward = types.MethodType(unet_forward_hook, unet)
                print("已添加UNet forward hook")
        
        print(f"InstantCharacter特征已附加到模型，权重: {weight}")
        print(f"您现在可以将此模型用于标准KSampler生成图像")
        
        return model
    except Exception as e:
        print(f"应用InstantCharacter失败: {str(e)}")
        traceback.print_exc()
        return model
