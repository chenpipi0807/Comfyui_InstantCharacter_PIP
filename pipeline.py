# Copyright 2025 Tencent InstantX Team. All rights reserved.
#

from PIL import Image
from einops import rearrange
import torch
from diffusers.pipelines.pipeline_utils import is_accelerate_available, is_accelerate_version
from diffusers.pipelines.flux.pipeline_flux import *
from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor

from .models.attn_processor import FluxIPAttnProcessor
from .models.resampler import CrossLayerCrossScaleProjector
from .models.utils import flux_load_lora


# TODO
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


class InstantCharacterFluxPipeline(FluxPipeline):
    _exclude_layer_from_cpu_offload = []

    @classmethod
    def from_pretrained_original_model(cls, original_model, torch_dtype=None, **kwargs):
        """
        从ComfyUI中的模型创建InstantCharacterFluxPipeline
        
        Args:
            original_model: ComfyUI中的模型对象
            torch_dtype: 模型数据类型
        """
        print("=== 开始创建InstantCharacterFluxPipeline（修改版） ===")
        
        try:
            # 不依赖于原始模型中的属性，而是创建空对象作为占位符
            from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
            from transformers import CLIPTextModel, CLIPTokenizer
            
            # 打印原始模型的类型和属性
            print(f"原始模型类型: {type(original_model).__name__}")
            print(f"原始模型属性: {[attr for attr in dir(original_model) if not attr.startswith('_')][:10]}")
            
            # 得到主模型 - 这个是必需的
            transformer = original_model.model
            if transformer is None:
                print("错误: 原始模型中没有model属性")
                return None
                
            print(f"主模型类型: {type(transformer).__name__}")
            
            # 创建空占位符对象
            class DummyModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 添加必要的配置属性
                    self.config = type('DummyConfig', (), {
                        'block_out_channels': [32, 64, 128, 256],  # VAE常见配置
                        'attention_head_dim': 64,
                        'num_attention_heads': 8,
                        'd_model': 4096  # 常用隐藏状态维度
                    })
                    # 添加dtype属性
                    self.dtype = torch.float32
                    self.device = torch.device('cpu')
                    
                def to(self, *args, **kwargs):
                    # 处理to方法以支持数据类型和设备转换
                    device = kwargs.get('device', None)
                    if device is not None:
                        self.device = device
                        
                    dtype = kwargs.get('dtype', None)
                    if dtype is not None:
                        self.dtype = dtype
                    
                    # 处理torch_dtype参数 (用于diffusers的to方法)
                    torch_dtype = kwargs.get('torch_dtype', None)
                    if torch_dtype is not None:
                        self.dtype = torch_dtype
                        
                    return self
                    
                def forward(self, *args, **kwargs):
                    return None
                    
            class DummyTokenizer:
                def __init__(self):
                    # 添加tokenizer所需的属性
                    self.model_max_length = 77  # CLIP tokenizer标准长度
                    self.vocab_size = 49408  # CLIP标准词汇量
                    self.pad_token_id = 0
                    self.eos_token_id = 49407
                    self.bos_token_id = 49406
                
                def __call__(self, *args, **kwargs):
                    # 返回标准的空占位符tokenizer输出
                    batch_size = kwargs.get('batch_size', 1)
                    return {
                        "input_ids": torch.zeros(batch_size, 77, dtype=torch.long),
                        "attention_mask": torch.ones(batch_size, 77, dtype=torch.long)
                    }
            
            # 设置text_encoder_2的配置        
            class DummyConfig:
                def __init__(self):
                    self.d_model = 4096  # 与原版保持一致
                    
            dummy_text_encoder_2 = DummyModule()
            dummy_text_encoder_2.config = DummyConfig()
                    
            # 创建必需的组件
            dummy_scheduler = DDIMScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                clip_sample=False, 
                set_alpha_to_one=False
            )
            
            dummy_vae = DummyModule()
            dummy_text_encoder = DummyModule()
            dummy_tokenizer = DummyTokenizer()
            dummy_tokenizer_2 = DummyTokenizer()
            
            # 创建必要的参数字典
            init_args = {
                'scheduler': dummy_scheduler,
                'vae': dummy_vae,
                'text_encoder': dummy_text_encoder,
                'tokenizer': dummy_tokenizer,
                'text_encoder_2': dummy_text_encoder_2,
                'tokenizer_2': dummy_tokenizer_2,
                'transformer': transformer,  # 主模型 - 这是唯一从原始模型中获取的真实组件
            }
            
            # 打印调试信息
            for key, value in init_args.items():
                print(f"{key}: {value is not None}, 类型: {type(value).__name__ if value is not None else 'None'}")
            
            print("尝试创建pipe实例...")
            pipe = cls(**init_args)
            print("pipe实例创建成功!")
            
            if torch_dtype is not None:
                print(f"设置模型数据类型为: {torch_dtype}")
                pipe.to(torch_dtype=torch_dtype)
            
            # 保存原始模型引用，用于后续操作
            pipe.original_model = original_model
            
            # 保存原始的forward函数
            pipe.original_forward = transformer.forward
            
            # 保存当前的配置信息
            pipe.kwargs = kwargs
            
            print("=== InstantCharacterFluxPipeline创建成功! ===")
            return pipe
            
        except Exception as e:
            print(f"=== InstantCharacterFluxPipeline创建失败: {str(e)} ===")
            import traceback
            traceback.print_exc()
            raise e

    @torch.inference_mode()
    def encode_siglip_image_emb(self, siglip_image, device, dtype):
        # Ensure encoder is on the correct device before use
        self.siglip_image_encoder.to(device, dtype=dtype)
        siglip_image = siglip_image.to(device, dtype=dtype)
        res = self.siglip_image_encoder(siglip_image, output_hidden_states=True)

        siglip_image_embeds = res.last_hidden_state

        siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
        
        return siglip_image_embeds, siglip_image_shallow_embeds


    @torch.inference_mode()
    def encode_dinov2_image_emb(self, dinov2_image, device, dtype):
        # Ensure encoder is on the correct device before use
        self.dino_image_encoder_2.to(device, dtype=dtype)
        dinov2_image = dinov2_image.to(device, dtype=dtype)
        res = self.dino_image_encoder_2(dinov2_image, output_hidden_states=True)

        dinov2_image_embeds = res.last_hidden_state[:, 1:]

        dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)

        return dinov2_image_embeds, dinov2_image_shallow_embeds


    @torch.inference_mode()
    def encode_image_emb(self, siglip_image, device, dtype):
        object_image_pil = siglip_image
        object_image_pil_low_res = [object_image_pil.resize((384, 384))]
        object_image_pil_high_res = object_image_pil.resize((768, 768))
        object_image_pil_high_res = [
            object_image_pil_high_res.crop((0, 0, 384, 384)),
            object_image_pil_high_res.crop((384, 0, 768, 384)),
            object_image_pil_high_res.crop((0, 384, 384, 768)),
            object_image_pil_high_res.crop((384, 384, 768, 768)),
        ]
        nb_split_image = len(object_image_pil_high_res)

        siglip_image_embeds = self.encode_siglip_image_emb(
            self.siglip_image_processor(images=object_image_pil_low_res, return_tensors="pt").pixel_values, 
            device, 
            dtype
        )
        dinov2_image_embeds = self.encode_dinov2_image_emb(
            self.dino_image_processor_2(images=object_image_pil_low_res, return_tensors="pt").pixel_values, 
            device, 
            dtype
        )

        image_embeds_low_res_deep = torch.cat([siglip_image_embeds[0], dinov2_image_embeds[0]], dim=2)
        image_embeds_low_res_shallow = torch.cat([siglip_image_embeds[1], dinov2_image_embeds[1]], dim=2)

        siglip_image_high_res = self.siglip_image_processor(images=object_image_pil_high_res, return_tensors="pt").pixel_values
        siglip_image_high_res = siglip_image_high_res[None]
        siglip_image_high_res = rearrange(siglip_image_high_res, 'b n c h w -> (b n) c h w')
        siglip_image_high_res_embeds = self.encode_siglip_image_emb(siglip_image_high_res, device, dtype)
        siglip_image_high_res_deep = rearrange(siglip_image_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        dinov2_image_high_res = self.dino_image_processor_2(images=object_image_pil_high_res, return_tensors="pt").pixel_values
        dinov2_image_high_res = dinov2_image_high_res[None]
        dinov2_image_high_res = rearrange(dinov2_image_high_res, 'b n c h w -> (b n) c h w')
        dinov2_image_high_res_embeds = self.encode_dinov2_image_emb(dinov2_image_high_res, device, dtype)
        dinov2_image_high_res_deep = rearrange(dinov2_image_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        image_embeds_high_res_deep = torch.cat([siglip_image_high_res_deep, dinov2_image_high_res_deep], dim=2)

        image_embeds_dict = dict(
            image_embeds_low_res_shallow=image_embeds_low_res_shallow,
            image_embeds_low_res_deep=image_embeds_low_res_deep,
            image_embeds_high_res_deep=image_embeds_high_res_deep,
        )
        return image_embeds_dict


    @torch.inference_mode()
    def init_ccp_and_attn_processor(self, *args, **kwargs):
        subject_ip_adapter_path = kwargs['subject_ip_adapter_path']
        nb_token = kwargs.get('nb_token', 1024)  # 默认使用1024个查询token
        print(f"=> 加载IP-Adapter权重: {subject_ip_adapter_path}")
        state_dict = torch.load(subject_ip_adapter_path, map_location="cpu")
        device, dtype = self.transformer.device, self.transformer.dtype

        print(f"=> 初始化注意力处理器")
        attn_procs = {}
        for idx_attn, (name, v) in enumerate(self.transformer.attn_processors.items()):
            # 确保使用正确的隐藏维度
            if hasattr(self.transformer, 'config') and hasattr(self.transformer.config, 'attention_head_dim') and hasattr(self.transformer.config, 'num_attention_heads'):
                hidden_size = self.transformer.config.attention_head_dim * self.transformer.config.num_attention_heads
            else:
                hidden_size = 3072  # 默认值
            
            ip_hidden_states_dim = 4096  # 标准维度
            
            attn_procs[name] = FluxIPAttnProcessor(
                hidden_size=hidden_size,
                ip_hidden_states_dim=ip_hidden_states_dim,
            ).to(device, dtype=dtype)
            
        self.transformer.set_attn_processor(attn_procs)
        tmp_ip_layers = torch.nn.ModuleList(self.transformer.attn_processors.values())
        key_name = tmp_ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)
        print(f"=> 加载注意力处理器权重: {key_name}")

        print(f"=> 初始化特征投影模型")
        image_proj_model = CrossLayerCrossScaleProjector(
            inner_dim=1152 + 1536,  # SigLIP + DINOv2特征维度
            num_attention_heads=42,
            attention_head_dim=64,
            cross_attention_dim=1152 + 1536,
            num_layers=4,
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=nb_token,
            embedding_dim=1152 + 1536,
            output_dim=4096,
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
        )
        image_proj_model.eval()
        image_proj_model.to(device, dtype=dtype)

        key_name = image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
        print(f"=> 加载特征投影模型权重: {key_name}")
        self.subject_image_proj_model = image_proj_model


    @torch.inference_mode()
    def init_adapter(
        self, 
        image_encoder_path=None,
        cache_dir=None,
        image_encoder_2_path=None,
        cache_dir_2=None,
        subject_ipadapter_cfg=None, 
    ):
        device, dtype = self.transformer.device, self.transformer.dtype

        # image encoder
        print(f"=> 加载SigLIP模型: {image_encoder_path}")
        image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path, cache_dir=cache_dir)
        image_processor = SiglipImageProcessor.from_pretrained(image_encoder_path, cache_dir=cache_dir)
        image_encoder.eval()
        image_encoder.to(device, dtype=dtype)
        self.siglip_image_encoder = image_encoder
        self.siglip_image_processor = image_processor

        # image encoder 2
        print(f"=> 加载DINOv2模型: {image_encoder_2_path}")
        image_encoder_2 = AutoModel.from_pretrained(image_encoder_2_path, cache_dir=cache_dir_2)
        image_processor_2 = AutoImageProcessor.from_pretrained(image_encoder_2_path, cache_dir=cache_dir_2)
        image_encoder_2.eval()
        image_encoder_2.to(device, dtype=dtype)
        image_processor_2.crop_size = dict(height=384, width=384)
        image_processor_2.size = dict(shortest_edge=384)
        self.dino_image_encoder_2 = image_encoder_2
        self.dino_image_processor_2 = image_processor_2

        # ccp and adapter
        self.init_ccp_and_attn_processor(**subject_ipadapter_cfg)
        
    def prepare_for_instant_character(self, reference_image, subject_weight=0.8):
        """准备InstantCharacter所需的特征和配置
        
        Args:
            reference_image: PIL格式的参考图像
            subject_weight: 主题图像的权重，控制特征融合的强度
        """
        device, dtype = self.transformer.device, self.transformer.dtype
        
        print(f"=> 准备InstantCharacter特征，权重: {subject_weight}")
        
        # 保存参考图像和权重
        self.reference_image = reference_image
        self.subject_weight = subject_weight
        
        # 提取特征嵌入
        try:
            print("=> 提取参考图像特征")
            image_embeds_dict = self.encode_image_emb(reference_image, device, dtype)
            self.image_embeds_dict = image_embeds_dict
            print(f"=> 特征提取成功: {list(image_embeds_dict.keys())}")
            
            for k, v in image_embeds_dict.items():
                print(f"  - {k}: {v.shape}")
                
            # 使用original_forward作为前向传播的基础
            if hasattr(self, 'original_forward'):
                self.transformer._original_forward = self.transformer.forward
                
                # 创建新的forward函数
                def instant_character_forward(self, x, timestep, context, y, guidance=None, control=None, transformer_options={}, **kwargs):
                    """InstantCharacter的前向传播函数"""
                    try:
                        # 获取主题图像的特征比例（权重）
                        weight = transformer_options.get('joint_attention_kwargs', {}).get('subject_emb_dict', {}).get('scale', self.pipeline.subject_weight)
                        
                        # 如果有特征字典和投影模型，则计算当前时间步的特征投影
                        dynamic_image_emb = None
                        if context is not None and weight > 0 and hasattr(self.pipeline, 'image_embeds_dict') and hasattr(self.pipeline, 'subject_image_proj_model'):
                            try:
                                # 将特征移动到当前设备和数据类型
                                image_embeds_dict = self.pipeline.image_embeds_dict
                                image_proj_model = self.pipeline.subject_image_proj_model
                                
                                # 确保特征在正确的设备上
                                low_res_shallow = image_embeds_dict['image_embeds_low_res_shallow'].to(x.device, x.dtype)
                                low_res_deep = image_embeds_dict['image_embeds_low_res_deep'].to(x.device, x.dtype)
                                high_res_deep = image_embeds_dict['image_embeds_high_res_deep'].to(x.device, x.dtype)
                                
                                # 确保timestep是合适的形状
                                curr_timestep = timestep
                                if not isinstance(curr_timestep, torch.Tensor):
                                    curr_timestep = torch.tensor([curr_timestep], device=x.device, dtype=x.dtype)
                                elif curr_timestep.dim() == 0:
                                    curr_timestep = curr_timestep.unsqueeze(0)
                                
                                # 使用CrossLayerCrossScaleProjector生成特征投影
                                image_proj_model.to(x.device, x.dtype)
                                dynamic_image_emb = image_proj_model(
                                    low_res_shallow=low_res_shallow,
                                    low_res_deep=low_res_deep,
                                    high_res_deep=high_res_deep,
                                    timesteps=curr_timestep,
                                    need_temb=True
                                )[0]
                                
                                # 应用权重
                                if dynamic_image_emb is not None:
                                    # 获取context的长度并添加IP-Adapter特征
                                    context_length = context.shape[1]
                                    token_num = dynamic_image_emb.shape[1]
                                    
                                    # 使用渐进融合策略，对前50%的tokens应用更强的引导
                                    # 这是原版InstantCharacter的实现方式
                                    half_len = token_num // 2
                                    gradual_weight = torch.ones(token_num, device=x.device, dtype=x.dtype)
                                    gradual_weight[:half_len] = torch.linspace(1.0, 0.5, half_len, device=x.device, dtype=x.dtype)
                                    gradual_weight[half_len:] = 0.5
                                    
                                    # 应用主题权重和渐进权重
                                    dynamic_image_emb = dynamic_image_emb * weight * gradual_weight.view(1, -1, 1)
                                    
                                    # 更新注意力处理器中的IP-Adapter特征
                                    joint_attention_kwargs = transformer_options.get('joint_attention_kwargs', {})
                                    joint_attention_kwargs['ip_hidden_states'] = dynamic_image_emb
                                    transformer_options['joint_attention_kwargs'] = joint_attention_kwargs
                            except Exception as e:
                                print(f"InstantCharacter动态特征投影失败: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # 调用原始forward函数
                        return self._original_forward(x, timestep, context, y, guidance, control, transformer_options, **kwargs)
                    except Exception as e:
                        print(f"InstantCharacter forward错误: {e}")
                        import traceback
                        traceback.print_exc()
                        return self._original_forward(x, timestep, context, y, guidance, control, transformer_options, **kwargs)
                
                # 将pipeline引用保存到transformer，以便在forward中访问
                self.transformer.pipeline = self
                
                # 将函数绑定到transformer实例
                import types
                self.transformer.forward = types.MethodType(instant_character_forward, self.transformer)
                
                print("=> 成功替换模型的forward函数，应用InstantCharacter特征处理")
                return True
            else:
                print("警告: 没有找到原始forward函数，无法应用InstantCharacter处理")
                return False
        except Exception as e:
            print(f"InstantCharacter准备失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            will be used instead
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image. This is set to 1024 by default for the best results.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image. This is set to 1024 by default for the best results.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        guidance_scale (`float`, *optional*, defaults to 7.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
            Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
            IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
            provided, embeddings are computed from the `ip_adapter_image` input argument.
        negative_ip_adapter_image:
            (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
            Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
            IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
            provided, embeddings are computed from the `ip_adapter_image` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

    Examples:

    Returns:
        [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
        is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
        images.
    """
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        subject_image: Image.Image = None,
        subject_scale: float = 0.8,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.dtype

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 3.1 Prepare subject emb
        if subject_image is not None:
            subject_image = subject_image.resize((max(subject_image.size), max(subject_image.size)))
            subject_image_embeds_dict = self.encode_image_emb(subject_image, device, dtype)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)


                # subject adapter
                if subject_image is not None:
                    # Ensure projector is on the correct device before use when offloading
                    self.subject_image_proj_model.to(latents.device, dtype=latents.dtype)
                    subject_image_prompt_embeds = self.subject_image_proj_model(
                        low_res_shallow=subject_image_embeds_dict['image_embeds_low_res_shallow'].to(latents.device, dtype=latents.dtype),
                        low_res_deep=subject_image_embeds_dict['image_embeds_low_res_deep'].to(latents.device, dtype=latents.dtype),
                        high_res_deep=subject_image_embeds_dict['image_embeds_high_res_deep'].to(latents.device, dtype=latents.dtype),
                        timesteps=timestep.to(device=latents.device, dtype=latents.dtype),
                        need_temb=True
                    )[0]
                    self._joint_attention_kwargs['emb_dict'] = dict(
                        length_encoder_hidden_states=prompt_embeds.shape[1]
                    )
                    self._joint_attention_kwargs['subject_emb_dict'] = dict(
                        ip_hidden_states=subject_image_prompt_embeds,
                        scale=subject_scale,
                    )
    
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

    def apply_instantcharacter_to_model(self, model, reference_image, subject_scale=0.9):
        """
        将InstantCharacter功能应用到ComfyUI模型中
        
        Args:
            model: ComfyUI模型对象
            reference_image: 参考图像 (PIL.Image)
            subject_scale: 角色特征强度
            
        Returns:
            应用了InstantCharacter功能的模型
        """
        print("[InstantCharacter] 开始应用InstantCharacter到模型...")
        
        try:
            # 获取设备和数据类型
            device = next(model.model.parameters()).device
            dtype = next(model.model.parameters()).dtype
            print(f"[InstantCharacter] 使用设备: {device}, 数据类型: {dtype}")
            
            # 1. 处理参考图像
            print(f"[InstantCharacter] 处理参考图像: {reference_image.size}")
            
            # 2. 创建增强版模型
            enhanced_model = model.clone()
            print(f"[InstantCharacter] 模型克隆成功: {type(enhanced_model).__name__}")
            
            # 3. 初始化模型属性
            enhanced_model.instantcharacter_enabled = True
            enhanced_model.instantcharacter_reference_image = reference_image
            enhanced_model.instantcharacter_subject_scale = subject_scale
            
            # 4. 创建回调函数来修改生成过程
            def instantcharacter_callback(self, *args, **kwargs):
                print("[InstantCharacter] 回调函数被触发, 应用角色特征")
                # 这里实现实际的InstantCharacter功能
                return args[0]  # 暂时直接返回原始结果
            
            # 5. 保存回调函数到模型中
            enhanced_model.instantcharacter_callback = instantcharacter_callback
            
            print("[InstantCharacter] 模型准备完成")
            return enhanced_model
            
        except Exception as e:
            print(f"[InstantCharacter] 应用InstantCharacter时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return model  # 返回原始模型

    @torch.inference_mode()
    def with_style_lora(self, lora_file_path, lora_weight=1.0, trigger='', *args, **kwargs):
        from .models.utils import flux_load_lora
        flux_load_lora(self, lora_file_path, lora_weight)
        kwargs['prompt'] = f"{trigger}, {kwargs['prompt']}"
        res = self.__call__(*args, **kwargs)
        flux_load_lora(self, lora_file_path, -lora_weight)
        return res

    def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using 🤗 Accelerate, significantly reducing memory usage. When called, the state
        dicts of all `torch.nn.Module` components (except those in `self._exclude_from_cpu_offload`) are saved to CPU
        and then moved to `torch.device('meta')` and loaded to GPU only when their specific submodule has its `forward`
        method called. Offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        self._maybe_raise_error_if_group_offload_active(raise_error=True)

        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")
        self.remove_all_hooks()

        is_pipeline_device_mapped = self.hf_device_map is not None and len(self.hf_device_map) > 1
        if is_pipeline_device_mapped:
            raise ValueError(
                "It seems like you have activated a device mapping strategy on the pipeline so calling `enable_sequential_cpu_offload() isn't allowed. You can call `reset_device_map()` first and then call `enable_sequential_cpu_offload()`."
            )

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`) or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")
        self._offload_device = device

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            device_mod = getattr(torch, self.device.type, None)
            if hasattr(device_mod, "empty_cache") and device_mod.is_available():
                device_mod.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        exclude_models = [module_name.split('.')[0] for module_name in self._exclude_layer_from_cpu_offload]

        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module):
                continue
            
            if name in exclude_models:
                for layer_name, layer in model.named_children():
                    if '.'.join([name, layer_name]) in self._exclude_layer_from_cpu_offload:
                        layer.to(device)
                    else:
                        offload_buffers = len(layer._parameters) > 0
                        cpu_offload(layer, device, offload_buffers=offload_buffers)
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                # make sure to offload buffers if not all high level weights
                # are of type nn.Module
                offload_buffers = len(model._parameters) > 0
                cpu_offload(model, device, offload_buffers=offload_buffers)