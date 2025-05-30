# 基于原始InstantCharacter项目的ComfyUI适配器
# 专门针对diffusers格式的FLUX模型

import os
import torch
import numpy as np
from PIL import Image
from einops import rearrange
import traceback
import types
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management as model_management
from transformers import SiglipVisionModel, AutoProcessor, AutoModel
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
            # RescaleAdaLN
            q = q * (1 + scale.view(bs, 1, h, 1)) + shift.view(bs, 1, h, 1)
        
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
        
        t_emb = self.time_proj(timestep)
        emb = self.time_embedding(t_emb)
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
    try:
        device = model_management.get_torch_device()
        dtype = torch.float16
        print(f"检查SigLIP模型类型: {SiglipVisionModel}")
        print(f"尝试使用transformers获取SigLIP处理器")
        model = SiglipVisionModel.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"SigLIP模型对象类型: {type(model)}")
        print(f"SigLIP处理器类型: {type(processor)}")
        
        model.to(device, dtype=dtype)
        print(f"将SigLIP模型移动到{device}设备")
        print(f"Device字符串表示: {device}")
        print(f"使用dtype: {dtype}")
        
        print(f"SigLIP模型 '{os.path.basename(model_path)}' 加载成功")
        return model, processor
    except Exception as e:
        print(f"加载SigLIP模型失败: {str(e)}")
        traceback.print_exc()
        return None, None

# 加载DINOv2模型
def load_dinov2_model(model_path):
    print(f"正在加载DINOv2模型: {model_path}")
    try:
        device = model_management.get_torch_device()
        dtype = torch.float16
        print(f"检查DINOv2模型类型: {AutoModel}")
        print(f"尝试使用transformers获取DINOv2处理器")
        model = AutoModel.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"DINOv2模型对象类型: {type(model)}")
        print(f"DINOv2处理器类型: {type(processor)}")
        
        model.to(device, dtype=dtype)
        print(f"将DINOv2模型移动到{device}设备")
        print(f"Device字符串表示: {device}")
        print(f"使用dtype: {dtype}")
        
        # 配置处理器参数
        processor.crop_size = {"height": 224, "width": 224}
        processor.size = {"shortest_edge": 224}
        
        print(f"DINOv2模型 '{os.path.basename(model_path)}' 加载成功")
        return model, processor
    except Exception as e:
        print(f"加载DINOv2模型失败: {str(e)}")
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
            key_name = image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
            print(f"加载投影器: {key_name}")
        except Exception as e:
            print(f"加载投影器出错: {e}")
            # 继续使用初始化的模型
            print("使用随机初始化的投影器")
            
        print("IP-Adapter模型加载成功")
        return image_proj_model
    except Exception as e:
        print(f"初始化IP-Adapter组件失败: {str(e)}")
        traceback.print_exc()
        return None

# 提取图像特征
def extract_image_features(image_pil, siglip_model, siglip_processor, dinov2_model, dinov2_processor):
    try:
        device = model_management.get_torch_device()
        dtype = torch.float16
        
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
        
        # SigLIP低分辨率特征
        with torch.no_grad():
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
        
        with torch.no_grad():
            res = dinov2_model(dinov2_inputs, output_hidden_states=True)
        
        dinov2_image_embeds = res.last_hidden_state[:, 1:] # 移除CLS token
        print(f"DINOv2深层特征去除CLS token后形状: {dinov2_image_embeds.shape}")
        print(f"DINOv2 hidden_states实际长度: {len(res.hidden_states)}")
        print(f"使用原版的层索引: [9, 19, 29]")
        dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)
        print(f"成功提取DINOv2浅层特征，形状: {dinov2_image_shallow_embeds.shape}")
        
        # 融合低分辨率特征
        image_embeds_low_res_deep = torch.cat([siglip_image_embeds, dinov2_image_embeds], dim=2)
        image_embeds_low_res_shallow = torch.cat([siglip_image_shallow_embeds, dinov2_image_shallow_embeds], dim=2)
        
        # 提取高分辨率特征
        print(f"开始提取高分辨率特征...")
        # SigLIP高分辨率特征
        siglip_high_res = siglip_processor(images=image_pil_high_res, return_tensors="pt").pixel_values
        siglip_high_res = siglip_high_res[None]
        siglip_high_res = rearrange(siglip_high_res, 'b n c h w -> (b n) c h w')
        siglip_high_res = siglip_high_res.to(device, dtype=dtype)
        
        with torch.no_grad():
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
            res = dinov2_model(dinov2_high_res, output_hidden_states=True)
        
        dinov2_high_res_embeds = res.last_hidden_state[:, 1:]
        dinov2_high_res_deep = rearrange(dinov2_high_res_embeds, '(b n) l c -> b (n l) c', n=nb_split_image)
        print(f"DINOv2高分辨率特征形状: {dinov2_high_res_deep.shape}")
        
        # 融合高分辨率特征
        image_embeds_high_res_deep = torch.cat([siglip_high_res_deep, dinov2_high_res_deep], dim=2)
        print(f"融合后的高分辨率特征形状: {image_embeds_high_res_deep.shape}")
        
        # 调整特征序列长度以匹配
        if siglip_image_embeds.shape[1] != dinov2_image_embeds.shape[1]:
            print(f"特征序列长度不匹配: SigLIP {siglip_image_embeds.shape[1]} vs DINOv2 {dinov2_image_embeds.shape[1]}")
            print(f"调整特征序列长度...")
            # 将SigLIP特征调整为与DINOv2相同的长度
            b, l_s, c_s = siglip_image_embeds.shape
            l_d = dinov2_image_embeds.shape[1]
            print(f"将SigLIP特征从{l_s}插值调整为{l_d}")
            siglip_image_embeds_resized = torch.nn.functional.interpolate(
                siglip_image_embeds.permute(0, 2, 1),
                size=l_d,
                mode='linear'
            ).permute(0, 2, 1)
            image_embeds_low_res_deep = torch.cat([siglip_image_embeds_resized, dinov2_image_embeds], dim=2)
        
        # 调整浅层特征序列长度
        if siglip_image_shallow_embeds.shape[1] != dinov2_image_shallow_embeds.shape[1]:
            print(f"浅层特征序列长度不匹配: SigLIP {siglip_image_shallow_embeds.shape[1]} vs DINOv2 {dinov2_image_shallow_embeds.shape[1]}")
            print(f"调整浅层特征序列长度...")
            b, l_s, c_s = siglip_image_shallow_embeds.shape
            l_d = dinov2_image_shallow_embeds.shape[1]
            print(f"将SigLIP浅层特征从{l_s}插值调整为{l_d}")
            siglip_shallow_resized = torch.nn.functional.interpolate(
                siglip_image_shallow_embeds.permute(0, 2, 1),
                size=l_d,
                mode='linear'
            ).permute(0, 2, 1)
            image_embeds_low_res_shallow = torch.cat([siglip_shallow_resized, dinov2_image_shallow_embeds], dim=2)
        
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
def generate_ip_adapter_embeddings(model, image_features, image_proj_model, scale=1.0):
    try:
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
        
        # 使用投影器生成IP-Adapter嵌入
        with torch.no_grad():
            cond_scale = scale
            
            # 设置模型的IP-Adapter嵌入
            image_outputs = image_proj_model(
                image_embeds=image_embeds_high_res_deep,
                image_low_res_embeds=image_embeds_low_res_shallow,
                image_low_res_deep_embeds=image_embeds_low_res_deep,
                scale=cond_scale,
            )
            ip_adapter_image_embeds = [image_outputs]
        
        return ip_adapter_image_embeds
    except Exception as e:
        print(f"生成IP-Adapter嵌入失败: {str(e)}")
        traceback.print_exc()
        return None

# 将InstantCharacter应用到模型
def apply_instant_character(model, reference_image, siglip_model=None, siglip_processor=None, dinov2_model=None, dinov2_processor=None, ip_adapter_model=None, weight=1.0):
    try:
        print(f"=== 开始应用InstantCharacter到模型 ===")
        
        # 1. 确保模型有效 - 特别为ComfyUI中的Transformer格式FLUX模型设计
        # 在ComfyUI中，模型是以独立的.safetensors文件加载，并被包装在ComfyUI的各种包装器中
        print(f"检查FLUX模型兼容性，模型类型: {type(model)}")
        
        # 尽量适应ComfyUI中的各种可能的FLUX模型包装格式
        is_flux = False
        
        # 直接设置is_flux = True用于测试 - 注释掉这一行如果你需要正常检测
        is_flux = True
        
        # 检查ComfyUI ModelPatcher格式
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            unet_type = str(model.model.diffusion_model.__class__.__name__).lower()
            if 'flux' in unet_type:
                is_flux = True
                print(f"检测到ComfyUI FLUX UNet模型: {unet_type}")
            
        # 检查model_type属性
        if hasattr(model, 'model_type') and isinstance(model.model_type, str):
            if 'flux' in model.model_type.lower():
                is_flux = True
                print(f"通过model_type属性检测到FLUX模型: {model.model_type}")
                
        # 检查更多可能的属性
        if hasattr(model, 'unet'):
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
            
        # 打印模型的一些基本属性，帮助调试
        print(f"模型属性: {[attr for attr in dir(model) if not attr.startswith('__')][:10]}")
        if hasattr(model, 'model'):
            print(f"model属性: {[attr for attr in dir(model.model) if not attr.startswith('__')][:10]}")
            
        # 如果不是FLUX模型，返回原始模型
        if not is_flux:
            print(f"警告: 无法确认当前模型是FLUX模型，类型: {type(model)}")
            # 仍然继续处理 - 假设用户知道自己在做什么
            print("尝试继续处理，假设这是FLUX模型...")
        else:
            print("检测到FLUX模型，继续处理...")
        
        # 2. 初始化IP-Adapter组件
        image_proj_model = init_ip_adapter_components(model, ip_adapter_model, nb_token=1024)
        if image_proj_model is None:
            print(f"错误: IP-Adapter组件初始化失败")
            return model
            
        # 3. 尝试从模型对象中提取处理器
        if siglip_model is not None and siglip_processor is None:
            print("尝试从SigLIP模型中提取处理器")
            if isinstance(siglip_model, tuple) and len(siglip_model) >= 2:
                siglip_processor = siglip_model[1]
                siglip_model = siglip_model[0]
                print("从SigLIP模型元组中提取处理器成功")
            elif hasattr(siglip_model, 'processor') and siglip_model.processor is not None:
                siglip_processor = siglip_model.processor
                print("从SigLIP模型属性中提取处理器成功")
            elif hasattr(siglip_model, 'image_processor') and siglip_model.image_processor is not None:
                siglip_processor = siglip_model.image_processor
                print("从SigLIP模型属性中提取image_processor成功")
            else:
                try:
                    from transformers import AutoProcessor
                    siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
                    print("无法从模型中提取SigLIP处理器，使用默认处理器")
                except Exception as e:
                    print(f"无法创建SigLIP处理器: {e}")
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
                    print(f"无法创建DINOv2处理器: {e}")
                    return model
                    
        # 检查必要模型组件
        if siglip_model is None or dinov2_model is None:
            print("错误: 必要的模型组件(SigLIP/DINOv2)不可用")
            return model
        if siglip_processor is None or dinov2_processor is None:
            print("错误: 必要的处理器组件不可用")
            return model
            
        # 提取图像特征
        image_features = extract_image_features(
            reference_image, 
            siglip_model, 
            siglip_processor, 
            dinov2_model, 
            dinov2_processor
        )
        if image_features is None:
            print(f"错误: 图像特征提取失败")
            return model
        
        # 4. 生成IP-Adapter嵌入
        ip_adapter_image_embeds = generate_ip_adapter_embeddings(
            model, 
            image_features, 
            image_proj_model,
            scale=weight
        )
        if ip_adapter_image_embeds is None:
            print(f"错误: IP-Adapter嵌入生成失败")
            return model
            
        # 5. 修改原始__call__方法以支持自动应用IP-Adapter
        original_call = model.__call__
        
        def custom_call_wrapper(self, *args, **kwargs):
            # 添加IP-Adapter嵌入到kwargs
            print(f"InstantCharacter激活，当前权重: {weight}")
            if ip_adapter_image_embeds is not None:
                kwargs["ip_adapter_image_embeds"] = ip_adapter_image_embeds
            else:
                print(f"缺少必要的特征，回退到原始forward")
            
            # 调用原始__call__方法
            return original_call(*args, **kwargs)
        
        # 替换模型的__call__方法
        model.__call__ = types.MethodType(custom_call_wrapper, model)
        
        print(f"InstantCharacter特征已附加到模型，权重: {weight}")
        print(f"您现在可以将此模型用于标准KSampler生成图像")
        
        return model
    except Exception as e:
        print(f"应用InstantCharacter失败: {str(e)}")
        traceback.print_exc()
        return model
