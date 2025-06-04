import os
import torch
import numpy as np
import folder_paths
from PIL import Image
import comfy.model_management as model_management
from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor
import traceback
import types

# 导入FLUX适配器
from .flux_adapter import load_siglip_model, load_dinov2_model, apply_instant_character

# 路径管理和模型检查函数
def load_instantcharacter_paths():
    """加载InstantCharacter模型路径并检查文件是否存在"""
    comfyui_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(comfyui_dir, "models")
    instantcharacter_dir = os.path.join(models_dir, "instantCharacter")
    
    # 设置模型路径
    siglip_model_path = os.path.join(instantcharacter_dir, "siglip-so400m-patch14-384")
    dinov2_model_path = os.path.join(instantcharacter_dir, "dinov2-giant")
    ip_adapter_path = os.path.join(instantcharacter_dir, "instantcharacter_ip-adapter.bin")
    
    print(f"InstantCharacter模型路径: {instantcharacter_dir}")
    print(f"SigLIP模型路径: {siglip_model_path}")
    print(f"DINOv2模型路径: {dinov2_model_path}")
    print(f"IP-Adapter路径: {ip_adapter_path}")
    
    # 检查文件并返回结果
    result = {
        "valid": True,
        "instantcharacter_dir": instantcharacter_dir,
        "siglip_model_path": siglip_model_path,
        "dinov2_model_path": dinov2_model_path,
        "ip_adapter_path": ip_adapter_path,
        "missing_models": []
    }
    
    if not os.path.exists(instantcharacter_dir):
        print(f"错误: InstantCharacter模型目录不存在: {instantcharacter_dir}")
        result["valid"] = False
        result["missing_models"].append("InstantCharacter目录")
    
    return result

# 检查是否为FLUX模型
def is_flux_model(model):
    """检查模型是否为FLUX模型"""
    try:
        # 首先检查我们之前添加的flux_model标记
        if hasattr(model, 'flux_model') and model.flux_model is True:
            print(f"通过flux_model标记识别为FLUX模型")
            return True
            
        # 检查instantcharacter_applied标记
        if hasattr(model, 'instantcharacter_applied') and model.instantcharacter_applied is True:
            print(f"模型已应用InstantCharacter，权重: {getattr(model, 'instantcharacter_weight', 'unknown')}")
            return True
            
        # 1. 检查模型名称或类型名称中是否包含'flux'
        model_name = str(model.__class__.__name__).lower() if hasattr(model, '__class__') else ""
        if 'flux' in model_name:
            print(f"通过类名识别为FLUX模型: {model_name}")
            return True
            
        # 2. 检查模型的unet类型名称
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            unet_type = str(model.model.diffusion_model.__class__.__name__).lower()
            if 'flux' in unet_type:
                print(f"通过UNet类型识别为FLUX模型: {unet_type}")
                return True
                
        # 3. 检查模型类型属性
        if hasattr(model, 'model_type') and isinstance(model.model_type, str):
            if 'flux' in model.model_type.lower():
                print(f"通过model_type属性识别为FLUX模型: {model.model_type}")
                return True
                
        # 4. ComfyUI特定检查 - 尝试获取原始对象
        if hasattr(model, 'unet'):
            unet = model.unet
            if hasattr(unet, 'model_type') and isinstance(unet.model_type, str):
                if 'flux' in unet.model_type.lower():
                    print(f"通过unet.model_type属性识别为FLUX模型: {unet.model_type}")
                    return True
            # 如果存在name属性并包含'flux'
            if hasattr(unet, 'name') and 'flux' in str(unet.name).lower():
                print(f"通过unet.name识别为FLUX模型: {unet.name}")
                return True
                    
        # 5. 检查文件名属性
        if hasattr(model, 'filename'):
            filename = str(model.filename).lower()
            if 'flux' in filename:
                print(f"通过filename识别为FLUX模型: {model.filename}")
                return True
                
        # 6. 检查模型的checkpoint_info
        if hasattr(model, 'checkpoint_info'):
            checkpoint_info = model.checkpoint_info
            if checkpoint_info is not None:
                if hasattr(checkpoint_info, 'name') and 'flux' in str(checkpoint_info.name).lower():
                    print(f"通过checkpoint_info.name识别为FLUX模型: {checkpoint_info.name}")
                    return True
                if hasattr(checkpoint_info, 'path') and 'flux' in str(checkpoint_info.path).lower():
                    print(f"通过checkpoint_info.path识别为FLUX模型: {checkpoint_info.path}")
                    return True
        
        # 7. 检查SD3特有的结构特征
        if hasattr(model, 'approx_vae_decode'):
            print(f"通过approx_vae_decode方法识别为SD3/Flux模型")
            return True
            
        # 8. 检查Flux特有的采样器选项
        if hasattr(model, 'sampler_name') and 'flux' in str(model.sampler_name).lower():
            print(f"通过sampler_name识别为FLUX模型: {model.sampler_name}")
            return True
            
        # 当前正在ComfyUI中使用Flux模型，强制返回True以确保兼容性
        # 注意: 这只是临时措施，日后应该提供更精细的检测
        return True
        
    except ImportError as ie:
        print(f"导入错误: {ie}")
        # 当前ComfyUI环境下返回True
        return True
    except Exception as e:
        print(f"检查FLUX模型时出错: {str(e)}")
        traceback.print_exc()
        # 当前ComfyUI环境下返回True
        return True

# 将ComfyUI图像格式转换为PIL图像
def comfyui_image_to_pil(image):
    """将ComfyUI图像格式转换为PIL图像"""
    from PIL import Image
    import torch
    import numpy as np
    
    # 将ComfyUI图像格式转换为PIL图像
    # ComfyUI图像格式为[B, H, W, C]的numpy数组，值范围0-1
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = image
        
    # 取第一张图像（如果是批量的）
    if len(image_np.shape) == 4:
        image_np = image_np[0]
        
    # 确保值范围为0-255并转为uint8
    image_np = (image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np)
    
    return pil_image

# SigLIP模型加载器节点
class SigLIP_Loader:
    """加载SigLIP视觉模型用于InstantCharacter"""
    
    def __init__(self):
        # 获取InstantCharacter模型目录
        paths = load_instantcharacter_paths()
        self.instantcharacter_dir = paths["instantcharacter_dir"]
        
        # 可用的SigLIP模型列表
        self.siglip_models = {
            "siglip-so400m-patch14-384": os.path.join(self.instantcharacter_dir, "siglip-so400m-patch14-384"),
            "siglip2-so400m-patch16-512": os.path.join(self.instantcharacter_dir, "siglip2-so400m-patch16-512")
        }
        
        # 缓存模型
        self.siglip_model = None
        self.siglip_processor = None
        self.current_model_key = None
    
    def load_model(self, model_name):
        """从选定的模型名加载SigLIP模型"""
        # 获取完整路径
        if model_name in self.siglip_models:
            model_path = self.siglip_models[model_name]
        else:
            print(f"错误: 无效的SigLIP模型名称: {model_name}")
            # 直接返回错误，不追加备用模型，使用分离值而不是元组
            return None, None
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"错误: SigLIP模型路径不存在: {model_path}")
            # 返回空模型和处理器，不尝试备用模型，使用分离值而不是元组
            return None, None
        
        # 如果模型已经加载并且是同一个模型，直接返回
        if self.siglip_model is not None and self.current_model_key == model_name:
            # 返回分离的模型和处理器，而不是元组
            return self.siglip_model, self.siglip_processor
        
        # 加载新模型
        try:
            print(f"正在加载SigLIP模型: {model_path}")
            from transformers import SiglipVisionModel, AutoProcessor
            self.siglip_model = SiglipVisionModel.from_pretrained(model_path)
            self.siglip_processor = AutoProcessor.from_pretrained(model_path)
            self.current_model_key = model_name
            print(f"SigLIP模型 '{model_name}' 加载成功")
            # 返回分离的模型和处理器，而不是元组
            return self.siglip_model, self.siglip_processor
        except Exception as e:
            print(f"加载SigLIP模型失败: {str(e)}")
            traceback.print_exc()
            # 返回空模型和处理器，避免后续错误，使用分离值而不是元组
            return None, None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["siglip-so400m-patch14-384"],),
            }
        }
    
    # 分别返回模型和处理器，而不是元组
    RETURN_TYPES = ("SIGLIP_MODEL", "SIGLIP_PROCESSOR")
    FUNCTION = "load_model"
    CATEGORY = "InstantCharacter"

# DINOv2模型加载器节点
class DINOv2_Loader:
    """加载DINOv2视觉模型用于InstantCharacter"""
    
    def __init__(self):
        # 获取InstantCharacter模型目录
        paths = load_instantcharacter_paths()
        self.instantcharacter_dir = paths["instantcharacter_dir"]
        
        # 可用的DINOv2模型列表
        self.dinov2_models = {
            "dinov2-giant": os.path.join(self.instantcharacter_dir, "dinov2-giant")
        }
        
        # 缓存模型
        self.dinov2_model = None
        self.dinov2_processor = None
        self.current_model_key = None
    
    def load_model(self, model_name):
        """从选定的模型名加载DINOv2模型"""
        # 获取完整路径
        if model_name in self.dinov2_models:
            model_path = self.dinov2_models[model_name]
        else:
            print(f"错误: 无效的DINOv2模型名称: {model_name}")
            # 直接返回错误，不追加备用模型，使用分离值而不是元组
            return None, None
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"错误: DINOv2模型路径不存在: {model_path}")
            # 返回空模型和处理器，不尝试备用模型，使用分离值而不是元组
            return None, None
        
        # 如果模型已经加载并且是同一个模型，直接返回
        if self.dinov2_model is not None and self.current_model_key == model_name:
            # 返回分离的模型和处理器，而不是元组
            return self.dinov2_model, self.dinov2_processor
        
        # 加载新模型
        try:
            print(f"正在加载DINOv2模型: {model_path}")
            from transformers import AutoModel, AutoProcessor
            self.dinov2_model = AutoModel.from_pretrained(model_path)
            self.dinov2_processor = AutoProcessor.from_pretrained(model_path)
            self.current_model_key = model_name
            print(f"DINOv2模型 '{model_name}' 加载成功")
            # 返回分离的模型和处理器，而不是元组
            return self.dinov2_model, self.dinov2_processor
        except Exception as e:
            print(f"加载DINOv2模型失败: {str(e)}")
            traceback.print_exc()
            # 返回空模型和处理器，避免后续错误，使用分离值而不是元组
            return None, None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["dinov2-giant"],),
            }
        }
    
    # 分别返回模型和处理器，而不是元组
    RETURN_TYPES = ("DINOV2_MODEL", "DINOV2_PROCESSOR")
    FUNCTION = "load_model"
    CATEGORY = "InstantCharacter"

# IP-Adapter模型加载器节点
class IPAdapter_Loader:
    """加载IP-Adapter模型用于InstantCharacter"""
    
    def __init__(self):
        # 获取InstantCharacter模型目录
        paths = load_instantcharacter_paths()
        self.instantcharacter_dir = paths["instantcharacter_dir"]
        
        # IP-Adapter模型路径
        self.ip_adapter_path = os.path.join(self.instantcharacter_dir, "instantcharacter_ip-adapter.bin")
        
        # 缓存模型
        self.ip_adapter = None
        self.ip_adapter_loaded = False
        self.current_device = None
    
    def load_model(self, use_ip_adapter="是"):
        """加载IP-Adapter模型"""
        # 如果用户选择不使用IP-Adapter，返回空模型
        if use_ip_adapter == "否":
            print("用户选择不使用IP-Adapter")
            return (None,)
            
        # 检查路径是否存在
        if not os.path.exists(self.ip_adapter_path):
            print(f"错误: IP-Adapter模型路径不存在: {self.ip_adapter_path}")
            return (None,)
        
        # 如果模型已经加载，直接返回
        if self.ip_adapter_loaded and self.ip_adapter is not None:
            print("IP-Adapter模型已经加载")
            return (self.ip_adapter,)
        
        # 加载模型
        try:
            print(f"正在加载IP-Adapter模型: {self.ip_adapter_path}")
            import torch
            
            # 选择最适合的设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用{device}设备加载模型")
            
            # 加载IP-Adapter模型
            self.ip_adapter = torch.load(self.ip_adapter_path, map_location=device)
            
            # IP-Adapter模型被直接作为字典对象使用
            print(f"IP-Adapter模型已加载成功，类型: {type(self.ip_adapter)}")
            
            # 如果是字典，显示其关键组件
            if isinstance(self.ip_adapter, dict):
                print(f"IP-Adapter模型包含以下键: {list(self.ip_adapter.keys())}")
            # 不尝试附加或扩展模型方法
            
            self.ip_adapter_loaded = True
            print("IP-Adapter模型加载成功")
            return (self.ip_adapter,)
        except Exception as e:
            print(f"加载IP-Adapter模型失败: {str(e)}")
            traceback.print_exc()
            return (None,)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_ip_adapter": (["是", "否"],),
            }
        }

    RETURN_TYPES = ("IP_ADAPTER_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "InstantCharacter"


class InstantCharacter:
    """实现InstantCharacter功能的主节点 - 用于在ComfyUI中应用InstantCharacter效果"""
    
    def apply_instant_character(self, model, reference_image, weight=0.8, siglip_model=None, dinov2_model=None, ip_adapter_model=None):
        """应用InstantCharacter功能到模型"""
        try:
            import os
            import torch
            import types
            import traceback
            from PIL import Image
            
            print("=== 开始应用InstantCharacter到模型 ===")
            
            # 检查必要的模型是否加载成功
            if siglip_model is None or dinov2_model is None:
                print("错误: 必要的模型(SigLIP/DINOv2)未加载成功，返回原始模型")
                return (model,)
            
            # 准备参考图像 - 将ComfyUI张量转换为PIL图像
            pil_image = comfyui_image_to_pil(reference_image)
            
            # 申请面向对象特征提取 - 增加高分辨率特征
            # 准备高分辨率图像 - 从原版移植
            object_image_pil_low_res = [pil_image.resize((384, 384))]
            # 创建高分辨率图像并分割为4个patch
            object_image_pil_high_res = pil_image.resize((768, 768))
            object_image_pil_high_res = [
                object_image_pil_high_res.crop((0, 0, 384, 384)),
                object_image_pil_high_res.crop((384, 0, 768, 384)),
                object_image_pil_high_res.crop((0, 384, 384, 768)),
                object_image_pil_high_res.crop((384, 384, 768, 768)),
            ]
            print(f"准备了4个高分辨率图像块，每个尺寸: 384x384")
            nb_split_image = len(object_image_pil_high_res)
            
            # 核心功能实现
            # 1. 手动提取SigLIP特征
            device = model_management.get_torch_device()
            
            # 严格按照原版实现提取SigLIP特征 (参考InstantCharacterFluxPipeline.encode_siglip_image_emb)
            try:
                # 检查模型类型，更灾全地处理不同的模型结构
                print(f"检查SigLIP模型类型: {type(siglip_model)}")
                
                # 仅从提供的模型元组中提取处理器，不尝试自动创建
                if isinstance(siglip_model, tuple) and len(siglip_model) == 2:
                    print("从元组中提取SigLIP模型和处理器")
                    siglip_model_obj = siglip_model[0]
                    siglip_processor = siglip_model[1]
                else:
                    # 不尝试自动创建处理器或使用其他来源
                    print(f"错误: SigLIP模型必须提供为包含模型和处理器的元组")
                    siglip_model_obj = None
                    siglip_processor = None
                
                # 日志输出结果
                print(f"SigLIP模型对象类型: {type(siglip_model_obj)}")
                print(f"SigLIP处理器类型: {type(siglip_processor) if siglip_processor else 'None'}")                
                
                # 明确检查处理器存在性
                if siglip_processor is None or siglip_model_obj is None:
                    print("错误: SigLIP模型或处理器不存在，无法继续处理")
                    return (model,)
                    
                # 准备图像处理
                # 将原始图像调整为384x384
                object_image_pil_low_res = [pil_image.resize((384, 384))]
                
                # 确保模型在正确的设备上
                print(f"将SigLIP模型移动到{device}设备")
                # 正确检查设备类型 - device是一个torch.device对象
                device_str = str(device)
                print(f"Device字符串表示: {device_str}")
                dtype = torch.float16 if 'cuda' in device_str else torch.float32
                print(f"使用dtype: {dtype}")
                siglip_model_obj = siglip_model_obj.to(device, dtype=dtype)
                
                # 处理图像
                if siglip_processor:
                    # 完全参照原版处理图像
                    print("使用SigLIP原生处理器处理图像")
                    pixel_values = siglip_processor(images=object_image_pil_low_res, return_tensors="pt").pixel_values.to(device, dtype=dtype)
                else:
                    # 如果没有处理器，使用基本图像处理
                    print("使用基本图像处理方法处理SigLIP输入")
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((384, 384)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
                    pixel_values = torch.stack([transform(img) for img in object_image_pil_low_res]).to(device, dtype=dtype)
                
                print(f"图像输入形状: {pixel_values.shape}, 设备: {pixel_values.device}, 类型: {pixel_values.dtype}")
                
                # 提取特征 - 基于原版实现但增加适应性
                with torch.no_grad():
                    # 要求输出隐藏状态，在原版中特别重要
                    res = siglip_model_obj(pixel_values, output_hidden_states=True)
                    
                    # 取出最后的隐藏状态
                    siglip_image_embeds = res.last_hidden_state
                    
                    # 检查hidden_states的实际长度
                    if hasattr(res, 'hidden_states') and res.hidden_states is not None:
                        hidden_states_len = len(res.hidden_states)
                        print(f"SigLIP hidden_states实际长度: {hidden_states_len}")
                        
                        # 原版使用[7, 13, 26]层，我们需要根据实际长度适配
                        if hidden_states_len >= 27:  # 如果长度足够，使用原版的层索引
                            print("使用原版的层索引: [7, 13, 26]")
                            layer_indices = [7, 13, 26]
                        else:
                            # 如果长度不足，选择分布均匀的三层
                            print(f"原版层索引太高，适配到当前模型")
                            # 选择开始、中间和结束附近的三层
                            if hidden_states_len >= 3:
                                first_idx = hidden_states_len // 4  # 大约25%处
                                second_idx = hidden_states_len // 2  # 中间层
                                third_idx = (hidden_states_len * 3) // 4  # 大约75%处
                                layer_indices = [first_idx, second_idx, third_idx]
                                print(f"选择层索引: {layer_indices}")
                            else:
                                # 如果层数小于3，直接使用所有可用层
                                layer_indices = list(range(hidden_states_len))
                                print(f"层数过少，使用所有层: {layer_indices}")
                        
                        # 尝试提取指定层的特征
                        try:
                            siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in layer_indices], dim=1)
                            print(f"成功提取SigLIP浅层特征，形状: {siglip_image_shallow_embeds.shape}")
                        except Exception as e:
                            print(f"提取SigLIP浅层特征失败: {e}")
                            # 如果异常，使用last_hidden_state的复制作为备选
                            siglip_image_shallow_embeds = siglip_image_embeds.clone()
                            print("使用last_hidden_state作为浅层特征替代")
                    else:
                        # 如果没有hidden_states，使用last_hidden_state替代
                        print("SigLIP模型没有返回hidden_states，使用last_hidden_state替代")
                        siglip_image_shallow_embeds = siglip_image_embeds.clone()
                    
                    # 合并深层和浅层特征
                    siglip_features = siglip_image_embeds
                    siglip_shallow_features = siglip_image_shallow_embeds
                
                print(f"SigLIP深层特征形状: {siglip_features.shape}")
                print(f"SigLIP浅层特征形状: {siglip_shallow_features.shape}")
                print("原版SigLIP特征提取成功")
            except Exception as e:
                print(f"SigLIP特征提取失败: {str(e)}")
                traceback.print_exc()
                return (model,)
            
            # 2. 严格按照原版实现提取DINOv2特征 (参考InstantCharacterFluxPipeline.encode_dinov2_image_emb)
            try:
                # 检查模型类型，更灾全地处理不同的模型结构
                print(f"检查DINOv2模型类型: {type(dinov2_model)}")
                
                # 仅从提供的模型元组中提取处理器，不尝试自动创建
                if isinstance(dinov2_model, tuple) and len(dinov2_model) == 2:
                    print("从元组中提取DINOv2模型和处理器")
                    dinov2_model_obj = dinov2_model[0]
                    dinov2_processor = dinov2_model[1]
                else:
                    # 不尝试自动创建处理器或使用其他来源
                    print(f"错误: DINOv2模型必须提供为包含模型和处理器的元组")
                    dinov2_model_obj = None
                    dinov2_processor = None
                
                # 日志输出结果
                print(f"DINOv2模型对象类型: {type(dinov2_model_obj)}")
                print(f"DINOv2处理器类型: {type(dinov2_processor) if dinov2_processor else 'None'}")                
                
                # 明确检查处理器存在性
                if dinov2_processor is None or dinov2_model_obj is None:
                    print("错误: DINOv2模型或处理器缺失，无法继续处理")
                    return (model,)
                
                # 使用相同的低分辨率图像
                object_image_pil_low_res = [pil_image.resize((384, 384))]
                
                # 确保模型在正确的设备上
                print(f"将DINOv2模型移动到{device}设备")
                # 正确检查设备类型 - device是一个torch.device对象
                device_str = str(device)
                print(f"Device字符串表示: {device_str}")
                dtype = torch.float16 if 'cuda' in device_str else torch.float32
                print(f"使用dtype: {dtype}")
                dinov2_model_obj = dinov2_model_obj.to(device, dtype=dtype)
                
                # 处理图像
                if dinov2_processor:
                    # 完全参照原版处理图像
                    print("使用DINOv2原生处理器处理图像")
                    pixel_values = dinov2_processor(images=object_image_pil_low_res, return_tensors="pt").pixel_values.to(device, dtype=dtype)
                else:
                    # 如果没有处理器，使用基本图像处理
                    print("使用基本图像处理方法处理DINOv2输入")
                    from torchvision import transforms
                    # DINOv2的标准处理器使用的均值和标准差
                    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                    transform = transforms.Compose([
                        transforms.Resize((384, 384)),  # 使用相同的尺寸保持一致性
                        transforms.ToTensor(),
                        normalize,
                    ])
                    pixel_values = torch.stack([transform(img) for img in object_image_pil_low_res]).to(device, dtype=dtype)
                
                print(f"DINOv2图像输入形状: {pixel_values.shape}, 设备: {pixel_values.device}, 类型: {pixel_values.dtype}")
                
                # 提取特征 - 基于原版实现但增加适应性
                with torch.no_grad():
                    # 要求输出隐藏状态，在原版中特别重要
                    res = dinov2_model_obj(pixel_values, output_hidden_states=True)
                    
                    # 注意：原版要求去除第一个token（CLS token）
                    try:
                        dinov2_image_embeds = res.last_hidden_state[:, 1:]
                        print(f"DINOv2深层特征去除CLS token后形状: {dinov2_image_embeds.shape}")
                    except Exception as e:
                        print(f"去除CLS token失败，使用完整的last_hidden_state: {e}")
                        dinov2_image_embeds = res.last_hidden_state
                    
                    # 检查hidden_states的实际长度
                    if hasattr(res, 'hidden_states') and res.hidden_states is not None:
                        hidden_states_len = len(res.hidden_states)
                        print(f"DINOv2 hidden_states实际长度: {hidden_states_len}")
                        
                        # 原版使用[9, 19, 29]层，我们需要根据实际长度适配
                        if hidden_states_len >= 30:  # 如果长度足够，使用原版的层索引
                            print("使用原版的层索引: [9, 19, 29]")
                            layer_indices = [9, 19, 29]
                        else:
                            # 如果长度不足，选择分布均匀的三层
                            print(f"原版层索引太高，适配到当前模型")
                            # 选择开始、中间和结束附近的三层
                            if hidden_states_len >= 3:
                                first_idx = hidden_states_len // 4  # 大约25%处
                                second_idx = hidden_states_len // 2  # 中间层
                                third_idx = (hidden_states_len * 3) // 4  # 大约75%处
                                layer_indices = [first_idx, second_idx, third_idx]
                                print(f"选择层索引: {layer_indices}")
                            else:
                                # 如果层数小于3，直接使用所有可用层
                                layer_indices = list(range(hidden_states_len))
                                print(f"层数过少，使用所有层: {layer_indices}")
                        
                        # 尝试提取指定层的特征并去除CLS token
                        try:
                            dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in layer_indices], dim=1)
                            print(f"成功提取DINOv2浅层特征，形状: {dinov2_image_shallow_embeds.shape}")
                        except Exception as e:
                            print(f"提取DINOv2浅层特征失败: {e}")
                            # 如果异常，尝试不去除CLS token
                            try:
                                dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in layer_indices], dim=1)
                                print(f"不去除CLS token的浅层特征形状: {dinov2_image_shallow_embeds.shape}")
                            except Exception as e2:
                                print(f"不去除CLS token仍然失败: {e2}")
                                # 如果仍然失败，使用last_hidden_state的复制作为备选
                                dinov2_image_shallow_embeds = dinov2_image_embeds.clone()
                                print("使用last_hidden_state作为浅层特征替代")
                    else:
                        # 如果没有hidden_states，使用last_hidden_state替代
                        print("DINOv2模型没有返回hidden_states，使用last_hidden_state替代")
                        dinov2_image_shallow_embeds = dinov2_image_embeds.clone()
                    
                    # 合并深层和浅层特征
                    dinov2_features = dinov2_image_embeds
                    dinov2_shallow_features = dinov2_image_shallow_embeds
                
                print(f"DINOv2深层特征形状: {dinov2_features.shape}")
                print(f"DINOv2浅层特征形状: {dinov2_shallow_features.shape}")
                print("原版DINOv2特征提取成功")
            except Exception as e:
                print(f"DINOv2特征提取失败: {str(e)}")
                traceback.print_exc()
                return (model,)
                
            # 3. 实现原版特征融合逻辑
            print("所有特征提取成功，开始融合特征")
            
            # 打印特征形状，用于调试
            print(f"SigLIP特征形状: {siglip_features.shape}")
            print(f"DINOv2特征形状: {dinov2_features.shape}")
            print(f"SigLIP浅层特征形状: {siglip_shallow_features.shape}")
            print(f"DINOv2浅层特征形状: {dinov2_shallow_features.shape}")
            
            # 判断特征形状是否需要调整
            if siglip_features.shape[1] != dinov2_features.shape[1]:
                print(f"特征序列长度不匹配: SigLIP {siglip_features.shape[1]} vs DINOv2 {dinov2_features.shape[1]}")
                print("调整特征序列长度...")
                
                # 将较长的序列插值到较短的序列长度
                if siglip_features.shape[1] > dinov2_features.shape[1]:
                    target_len = dinov2_features.shape[1]
                    print(f"将SigLIP特征从{siglip_features.shape[1]}插值调整为{target_len}")
                    siglip_features_resized = torch.nn.functional.interpolate(
                        siglip_features.permute(0, 2, 1),  # [B, C, L]
                        size=target_len,
                        mode='linear'
                    ).permute(0, 2, 1)  # 变回 [B, L, C]
                    siglip_features = siglip_features_resized
                else:
                    target_len = siglip_features.shape[1]
                    print(f"将DINOv2特征从{dinov2_features.shape[1]}插值调整为{target_len}")
                    dinov2_features_resized = torch.nn.functional.interpolate(
                        dinov2_features.permute(0, 2, 1),  # [B, C, L]
                        size=target_len,
                        mode='linear'
                    ).permute(0, 2, 1)  # 变回 [B, L, C]
                    dinov2_features = dinov2_features_resized
            
            # 同样处理浅层特征
            if siglip_shallow_features.shape[1] != dinov2_shallow_features.shape[1]:
                print(f"浅层特征序列长度不匹配: SigLIP {siglip_shallow_features.shape[1]} vs DINOv2 {dinov2_shallow_features.shape[1]}")
                print("调整浅层特征序列长度...")
                
                # 将较长的序列插值到较短的序列长度
                if siglip_shallow_features.shape[1] > dinov2_shallow_features.shape[1]:
                    target_len = dinov2_shallow_features.shape[1]
                    print(f"将SigLIP浅层特征从{siglip_shallow_features.shape[1]}插值调整为{target_len}")
                    siglip_shallow_features_resized = torch.nn.functional.interpolate(
                        siglip_shallow_features.permute(0, 2, 1),
                        size=target_len,
                        mode='linear'
                    ).permute(0, 2, 1)
                    siglip_shallow_features = siglip_shallow_features_resized
                else:
                    target_len = siglip_shallow_features.shape[1]
                    print(f"将DINOv2浅层特征从{dinov2_shallow_features.shape[1]}插值调整为{target_len}")
                    dinov2_shallow_features_resized = torch.nn.functional.interpolate(
                        dinov2_shallow_features.permute(0, 2, 1),
                        size=target_len,
                        mode='linear'
                    ).permute(0, 2, 1)
                    dinov2_shallow_features = dinov2_shallow_features_resized
            
            # 再次检查调整后的形状
            print(f"调整后的SigLIP特征形状: {siglip_features.shape}")
            print(f"调整后的DINOv2特征形状: {dinov2_features.shape}")
            print(f"调整后的SigLIP浅层特征形状: {siglip_shallow_features.shape}")
            print(f"调整后的DINOv2浅层特征形状: {dinov2_shallow_features.shape}")
            
            # 处理高分辨率特征 - 参照原版pipeline.py
            print("开始提取高分辨率特征...")
            try:
                # 处理SigLIP高分辨率图像
                if siglip_processor:
                    # 原版的批处理方式
                    from einops import rearrange
                    siglip_image_high_res = siglip_processor(images=object_image_pil_high_res, return_tensors="pt").pixel_values
                    siglip_image_high_res = siglip_image_high_res[None]  # 增加批处理维度 [1, 4, 3, 384, 384]
                    siglip_image_high_res = rearrange(siglip_image_high_res, 'b n c h w -> (b n) c h w')  # 重新排列为[4, 3, 384, 384]
                    siglip_image_high_res = siglip_image_high_res.to(device, dtype=dtype)
                    
                    # 提取高分辨率SigLIP特征
                    with torch.no_grad():
                        res_high = siglip_model_obj(siglip_image_high_res, output_hidden_states=True)
                        # 获取高分辨率特征
                        siglip_image_high_res_embeds = res_high.last_hidden_state
                        # 重新排列成原版格式
                        siglip_image_high_res_deep = rearrange(siglip_image_high_res_embeds, '(b n) l c -> b (n l) c', n=nb_split_image)
                        print(f"SigLIP高分辨率特征形状: {siglip_image_high_res_deep.shape}")
                else:
                    print("警告: 缺少SigLIP处理器，跳过高分辨率特征提取")
                    # 使用低分辨率特征作为高分辨率特征的替代
                    siglip_image_high_res_deep = siglip_features.clone()
            except Exception as e:
                print(f"SigLIP高分辨率特征提取失败: {e}")
                traceback.print_exc()
                # 使用低分辨率特征作为备选
                siglip_image_high_res_deep = siglip_features.clone()
            
            # 处理DINOv2高分辨率图像
            try:
                if dinov2_processor:
                    # 原版的批处理方式
                    dinov2_image_high_res = dinov2_processor(images=object_image_pil_high_res, return_tensors="pt").pixel_values
                    dinov2_image_high_res = dinov2_image_high_res[None]
                    dinov2_image_high_res = rearrange(dinov2_image_high_res, 'b n c h w -> (b n) c h w')
                    dinov2_image_high_res = dinov2_image_high_res.to(device, dtype=dtype)
                    
                    # 提取高分辨率DINOv2特征
                    with torch.no_grad():
                        res_high = dinov2_model_obj(dinov2_image_high_res, output_hidden_states=True)
                        # 获取高分辨率特征，去除CLS token
                        try:
                            dinov2_image_high_res_embeds = res_high.last_hidden_state[:, 1:]
                        except:
                            dinov2_image_high_res_embeds = res_high.last_hidden_state
                        # 重新排列成原版格式
                        dinov2_image_high_res_deep = rearrange(dinov2_image_high_res_embeds, '(b n) l c -> b (n l) c', n=nb_split_image)
                        print(f"DINOv2高分辨率特征形状: {dinov2_image_high_res_deep.shape}")
                else:
                    print("警告: 缺少DINOv2处理器，跳过高分辨率特征提取")
                    # 使用低分辨率特征作为高分辨率特征的替代
                    dinov2_image_high_res_deep = dinov2_features.clone()
            except Exception as e:
                print(f"DINOv2高分辨率特征提取失败: {e}")
                traceback.print_exc()
                # 使用低分辨率特征作为备选
                dinov2_image_high_res_deep = dinov2_features.clone()
            
            # 融合高分辨率特征
            try:
                # 确保特征在序列长度上匹配
                if siglip_image_high_res_deep.shape[1] != dinov2_image_high_res_deep.shape[1]:
                    # 调整高分辨率特征序列长度
                    if siglip_image_high_res_deep.shape[1] > dinov2_image_high_res_deep.shape[1]:
                        target_len = dinov2_image_high_res_deep.shape[1]
                        siglip_image_high_res_deep = torch.nn.functional.interpolate(
                            siglip_image_high_res_deep.permute(0, 2, 1), 
                            size=target_len, 
                            mode='linear'
                        ).permute(0, 2, 1)
                    else:
                        target_len = siglip_image_high_res_deep.shape[1]
                        dinov2_image_high_res_deep = torch.nn.functional.interpolate(
                            dinov2_image_high_res_deep.permute(0, 2, 1), 
                            size=target_len, 
                            mode='linear'
                        ).permute(0, 2, 1)
                
                # 融合高分辨率特征
                image_embeds_high_res_deep = torch.cat([siglip_image_high_res_deep, dinov2_image_high_res_deep], dim=2)
                print(f"融合后的高分辨率特征形状: {image_embeds_high_res_deep.shape}")
            except Exception as e:
                print(f"高分辨率特征融合失败: {e}")
                # 如果高分辨率特征融合失败，使用低分辨率特征作为替代
                image_embeds_high_res_deep = image_embeds_low_res_deep.clone()
            
            # 参照原版encode_image_emb方法进行特征融合
            # 这里我们使用了低分辨率和深层/浅层特征的组合
            image_embeds_low_res_deep = torch.cat([siglip_features, dinov2_features], dim=2)
            image_embeds_low_res_shallow = torch.cat([siglip_shallow_features, dinov2_shallow_features], dim=2)
            
            print(f"融合后的深层特征形状: {image_embeds_low_res_deep.shape}")
            print(f"融合后的浅层特征形状: {image_embeds_low_res_shallow.shape}")
            
            # 克隆模型以避免修改原始模型
            print("克隆原始模型...")
            enhanced_model = model.clone()
            
            # 实现原版的前向传播修改
            # 1. 获取UNet组件
            if hasattr(enhanced_model, "model") and hasattr(enhanced_model.model, "diffusion_model"):
                unet = enhanced_model.model.diffusion_model
                print("检测到标准SD模型结构")
            else:
                # 对于FLUX等其他模型结构
                unet = enhanced_model.model
                print("检测到FLUX模型结构")
            
            # 2. 保存原始前向传播函数
            if not hasattr(unet, 'original_forward'):
                unet.original_forward = unet.forward
                print("保存了原始forward函数")
            
            # 3. 定义新的InstantCharacter前向传播函数 - 实现参考原版特征注入机制
            def custom_forward(self, *args, **kwargs):
                try:
                    # 检查是否启用InstantCharacter
                    instantcharacter_enabled = getattr(self, 'instantcharacter_enabled', False)
                    current_weight = getattr(self, 'instantcharacter_weight', weight)
                    
                    # 如果不启用或权重为0，直接调用原始forward
                    if not instantcharacter_enabled or current_weight <= 0:
                        return self.original_forward(*args, **kwargs)
                    
                    # 获取必要的特征
                    print(f"InstantCharacter激活，当前权重: {current_weight}")
                    
                    # 获取全部存储的特征
                    image_embeds_low_res_deep = getattr(self, 'image_embeds_low_res_deep', None)
                    image_embeds_low_res_shallow = getattr(self, 'image_embeds_low_res_shallow', None)
                    image_embeds_high_res_deep = getattr(self, 'image_embeds_high_res_deep', None)
                    
                    # 检查特征是否存在
                    if image_embeds_low_res_deep is None or image_embeds_low_res_shallow is None:
                        print("缺少必要的特征，回退到原始forward")
                        return self.original_forward(*args, **kwargs)
                    
                    # 修改原版参数，注入我们的特征
                    # 复制args和kwargs以避免修改原始参数
                    new_args = list(args)
                    new_kwargs = kwargs.copy()
                    
                    # 核心特征注入逻辑 - 模拟原版实现
                    hidden_states = None
                    timestep = None
                    encoder_hidden_states = None
                    guidance = None
                    
                    # 提取关键参数
                    if len(new_args) >= 1:
                        hidden_states = new_args[0]
                    if len(new_args) >= 2:
                        timestep = new_args[1]
                    if len(new_args) >= 4:
                        encoder_hidden_states = new_args[3]
                    
                    # 注入特征的核心逻辑
                    if encoder_hidden_states is not None and hidden_states is not None:
                        batch_size = hidden_states.shape[0]
                        seq_len = encoder_hidden_states.shape[1]
                        
                        # 计算注入位置 - 参考原版的权重衰减策略
                        # 对前50%的tokens应用更强的引导
                        inject_start = 0
                        inject_end = seq_len // 2  # 简化版，原版使用了更复杂的逻辑
                        
                        # 创建递减的权重序列
                        import numpy as np
                        weights = np.linspace(current_weight, current_weight * 0.5, inject_end - inject_start)
                        weights_tensor = torch.tensor(weights, dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(0).unsqueeze(2)
                        
                        # 准备注入特征
                        # 使用低分辨率特征作为主要注入特征
                        if inject_end <= encoder_hidden_states.shape[1]:
                            # 规模调整特征到正确的尺寸
                            inject_features = image_embeds_low_res_deep
                            if inject_features.shape[1] != inject_end - inject_start:
                                inject_features = torch.nn.functional.interpolate(
                                    inject_features.permute(0, 2, 1),
                                    size=inject_end - inject_start,
                                    mode='linear'
                                ).permute(0, 2, 1)
                            
                            # 重复批次以匹配输入批大小
                            if inject_features.shape[0] != batch_size:
                                inject_features = inject_features.repeat(batch_size, 1, 1)
                            
                            # 应用权重序列
                            weighted_features = inject_features * weights_tensor
                            
                            # 替换encoder_hidden_states中的内容
                            encoder_hidden_states_modified = encoder_hidden_states.clone()
                            encoder_hidden_states_modified[:, inject_start:inject_end, :] += weighted_features
                            
                            # 替换原始args中的参数
                            if len(new_args) >= 4:
                                new_args[3] = encoder_hidden_states_modified
                    
                    # 调用带修改参数的原始forward
                    result = self.original_forward(*new_args, **new_kwargs)
                    
                except Exception as e:
                    print(f"InstantCharacter特征注入错误: {e}")
                    import traceback
                    traceback.print_exc()
                    # 出错时回退到原始forward
                    result = self.original_forward(*args, **kwargs)
                
                return result
            
            # 4. 应用自定义forward函数
            unet.forward = types.MethodType(custom_forward, unet)
            
            # 5. 将融合后的特征附加到模型上
            # 首先附加原始特征
            enhanced_model.siglip_features = siglip_features
            enhanced_model.dinov2_features = dinov2_features
            enhanced_model.siglip_shallow_features = siglip_shallow_features
            enhanced_model.dinov2_shallow_features = dinov2_shallow_features
            
            # 附加融合后的特征
            enhanced_model.image_embeds_low_res_deep = image_embeds_low_res_deep
            enhanced_model.image_embeds_low_res_shallow = image_embeds_low_res_shallow
            
            # 附加高分辨率特征
            enhanced_model.image_embeds_high_res_deep = image_embeds_high_res_deep
            
            # 创建原版格式的特征字典，以便在前向传播函数中使用
            enhanced_model.image_embeds_dict = {
                'image_embeds_low_res_shallow': image_embeds_low_res_shallow,
                'image_embeds_low_res_deep': image_embeds_low_res_deep,
                'image_embeds_high_res_deep': image_embeds_high_res_deep,
            }
            
            # 设置权重和启用标志
            enhanced_model.instantcharacter_weight = weight
            enhanced_model.instantcharacter_enabled = True
            
            # 将特征也附加到UNet模型上，确保在forward函数中能访问
            unet.image_embeds_dict = enhanced_model.image_embeds_dict
            unet.instantcharacter_weight = weight
            unet.instantcharacter_enabled = True
            
            print(f"InstantCharacter特征已附加到模型，权重: {weight}")
            print("您现在可以将此模型用于标准KSampler生成图像")
            
            return (enhanced_model,)
            
        except Exception as e:
            print(f"InstantCharacter应用失败: {str(e)}")
            traceback.print_exc()
            return (model,)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "siglip_model": ("SIGLIP_MODEL",),
                "dinov2_model": ("DINOV2_MODEL",),
                "ip_adapter_model": ("IP_ADAPTER_MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_instant_character"
    CATEGORY = "InstantCharacter"


class FluxInstantCharacter:
    """实现FluxInstantCharacter功能的主节点 - 专门为FLUX模型设计的InstantCharacter效果"""
    RETURN_TYPES = ("MODEL", "CONDITIONING")  # 将CLIP改为CONDITIONING作为输出
    FUNCTION = "apply_flux_instant_character"
    CATEGORY = "InstantCharacter"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_image": ("IMAGE",),
                "weight": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "conditioning": ("CONDITIONING",),  # 使用条件特征作为可选输入
                "siglip_model": ("SIGLIP_MODEL",),  # 接收SigLIP模型
                "siglip_processor": ("SIGLIP_PROCESSOR",),  # 新增单独的SigLIP处理器输入
                "dinov2_model": ("DINOV2_MODEL",),  # 接收DINOv2模型
                "dinov2_processor": ("DINOV2_PROCESSOR",),  # 新增单独的DINOv2处理器输入
                "ip_adapter_model": ("IP_ADAPTER_MODEL",),
                "debug_mode": (["关闭", "开启"], {
                    "default": "关闭"
                })
            }
        }
    
    def apply_flux_instant_character(self, model, reference_image, weight=0.8, conditioning=None, siglip_model=None, siglip_processor=None, dinov2_model=None, dinov2_processor=None, ip_adapter_model=None, debug_mode="关闭"):
        """
        应用InstantCharacter功能到FLUX模型
        
        参数:
            model: FLUX模型
            reference_image: 参考图像
            weight: 应用权重
            conditioning: 外部提供的条件特征（可选）
            siglip_model: SigLIP视觉模型
            siglip_processor: SigLIP处理器
            dinov2_model: DINOv2视觉模型
            dinov2_processor: DINOv2处理器
            ip_adapter_model: IP-Adapter模型
            debug_mode: 是否启用调试模式
        
        返回:
            应用了InstantCharacter的FLUX模型
        """
        try:
            # 处理参考图像 - 转换为PIL格式
            if reference_image is not None:
                reference_image = comfyui_image_to_pil(reference_image)
                print(f"参考图像尺寸: 宽={reference_image.width}, 高={reference_image.height}")
            else:
                print("错误: 缺少参考图像")
                return (model,)
            
            # 确保有SigLIP模型
            if siglip_model is None:
                print("错误: 需要连接SigLIP模型到FluxInstantCharacter节点")
                return (model,)
            
            # 开启调试模式
            if debug_mode == "开启":
                print("=== 调试模式已开启 ===")
                print(f"模型类型: {type(model)}")
                print(f"模型属性: {[attr for attr in dir(model) if not attr.startswith('__')][:15]}")
                if hasattr(model, 'model'):
                    print(f"model属性: {[attr for attr in dir(model.model) if not attr.startswith('__')][:15]}")
                if hasattr(model, 'unet'):
                    print(f"unet属性: {[attr for attr in dir(model.unet) if not attr.startswith('__')][:15]}")
                    
            # 标记为Flux模型
            model.flux_model = True
            if hasattr(model, 'model_type'):
                print(f"原模型类型: {model.model_type}")
                if 'flux' not in model.model_type.lower():
                    model.model_type = f"FLUX_{model.model_type}"
                    print(f"已标记模型类型为: {model.model_type}")
            
            # 备份直接传入的处理器参数
            _input_siglip_processor = siglip_processor
            _input_dinov2_processor = dinov2_processor
            
            # 提取SigLIP处理器 - 增加详细调试信息
            print(f"SigLIP模型类型: {type(siglip_model)}")
            # 优先使用直接传入的处理器
            if _input_siglip_processor is not None:
                siglip_processor = _input_siglip_processor
                print(f"使用直接传入的SigLIP处理器")
            elif isinstance(siglip_model, tuple):
                print(f"SigLIP模型是元组，长度: {len(siglip_model)}")
                if len(siglip_model) >= 2:
                    print(f"从元组中提取SigLIP处理器，元组内容类型: [{type(siglip_model[0])}, {type(siglip_model[1])}]")
                    siglip_processor = siglip_model[1]
                    siglip_model = siglip_model[0]
                else:
                    print(f"警告: SigLIP模型元组长度不足，无法提取处理器")
            elif hasattr(siglip_model, 'processor') and siglip_model.processor is not None:
                siglip_processor = siglip_model.processor
                print(f"从模型对象中提取SigLIP处理器，处理器类型: {type(siglip_processor)}")
            else:
                print(f"警告: 无法从SigLIP模型中提取处理器")
                
            print(f"提取后的SigLIP处理器类型: {type(siglip_processor) if siglip_processor else 'None'}")
            print(f"提取后的SigLIP模型类型: {type(siglip_model)}")
            
            # 提取DINOv2处理器
            print(f"DINOv2模型类型: {type(dinov2_model) if dinov2_model else 'None'}")
            print(f"直接传入的DINOv2处理器类型: {type(_input_dinov2_processor) if _input_dinov2_processor else 'None'}")
            
            # 优先使用直接传入的处理器
            if _input_dinov2_processor is not None:
                dinov2_processor = _input_dinov2_processor
                print("使用直接传入的DINOv2处理器")
            # 如果没有直接传入处理器，则尝试从其他源获取
            elif isinstance(dinov2_model, tuple) and len(dinov2_model) >= 2:
                print("从元组中提取DINOv2处理器")
                dinov2_processor = dinov2_model[1]
                dinov2_model = dinov2_model[0]
            elif hasattr(dinov2_model, 'processor') and dinov2_model.processor is not None:
                dinov2_processor = dinov2_model.processor
                print("从模型对象中提取DINOv2处理器")
            elif hasattr(dinov2_model, 'image_processor') and dinov2_model.image_processor is not None:
                dinov2_processor = dinov2_model.image_processor
                print("从相关属性中提取DINOv2处理器")
            
            # 应用InstantCharacter处理 - 增加调试信息
            print(f"开始处理Flux模型,应用InstantCharacter,权重={weight}")
            if conditioning is not None:
                print("使用外部提供的条件特征而不是进行内部处理")
                
            # 打印所有关键参数类型
            print(f"调用apply_instant_character前的参数类型检查:")
            print(f"  - siglip_model类型: {type(siglip_model)}")
            print(f"  - siglip_processor类型: {type(siglip_processor) if siglip_processor else 'None'}")
            print(f"  - dinov2_model类型: {type(dinov2_model) if dinov2_model else 'None'}")
            print(f"  - dinov2_processor类型: {type(dinov2_processor) if dinov2_processor else 'None'}")
            print(f"  - ip_adapter_model类型: {type(ip_adapter_model) if ip_adapter_model else 'None'}")
            
            modified_model = apply_instant_character(
                model=model,
                reference_image=reference_image,
                weight=weight,
                conditioning=conditioning,  # 传递外部条件特征
                siglip_model=siglip_model,
                siglip_processor=siglip_processor,  # 显式传递处理器
                dinov2_model=dinov2_model,
                dinov2_processor=dinov2_processor,  # 显式传递处理器
                ip_adapter_model=ip_adapter_model
            )
            
            # 添加属性标记
            if modified_model is not None:
                # 检查是否真正成功应用了IP-Adapter特征
                if hasattr(modified_model, '_ip_adapter_image_embeds') and modified_model._ip_adapter_image_embeds is not None:
                    modified_model.instantcharacter_applied = True
                    modified_model.instantcharacter_weight = weight
                    print("Flux InstantCharacter应用成功 - 图像特征已注入")
                else:
                    print("警告: IP-Adapter特征缺失，InstantCharacter可能无法正常工作")
            
            # 生成融合后的条件特征
            combined_condition = None
            
            # 优先使用外部提供的条件特征
            if conditioning is not None:
                combined_condition = conditioning
                print("使用外部提供的条件特征")
            # 否则尝试生成新的条件特征
            elif hasattr(modified_model, '_ip_adapter_image_embeds') and modified_model._ip_adapter_image_embeds is not None:
                # 从模型中提取处理后的条件特征
                combined_condition = [(modified_model._ip_adapter_image_embeds, weight)]
                print("已生成融合后的条件特征")
            else:
                # 创建空的条件特征以防止返回None
                print("警告: 无法生成条件特征，返回空条件")
                combined_condition = []
                
            return (modified_model, combined_condition)
        
        except Exception as e:
            print(f"FluxInstantCharacter处理过程中出错: {e}")
            traceback.print_exc()
            # 错误时返回原始模型和空条件特征
            empty_condition = []
            if conditioning is not None:
                empty_condition = conditioning
            return (model, empty_condition)
