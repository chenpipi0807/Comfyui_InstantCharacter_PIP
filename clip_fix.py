import torch
import os
import traceback
from typing import Optional, Dict, Any, Union

def fix_clip_model_missing_params(clip_model: Any) -> bool:
    """
    检查CLIP模型是否缺少text_projection.weight参数，如果缺少则创建一个适当的参数。
    
    Args:
        clip_model: CLIP模型对象
        
    Returns:
        bool: 是否成功修复了模型
    """
    try:
        print(f"检查CLIP模型参数完整性")
        if clip_model is None:
            print(f"CLIP模型为空，无法修复")
            return False
            
        # 检查模型是否为预期类型
        model_type = type(clip_model).__name__
        print(f"CLIP模型类型: {model_type}")
        
        # 检查是否有text_model属性
        has_text_model = hasattr(clip_model, 'text_model')
        print(f"是否有text_model属性: {has_text_model}")
        
        # 如果没有text_model，可能有其他结构
        if not has_text_model and hasattr(clip_model, 'transformer'):
            print(f"检测到transformer结构而非text_model")
            # 某些CLIP变体可能使用transformer代替text_model
            text_model = clip_model.transformer
        elif has_text_model:
            text_model = clip_model.text_model
        else:
            # 尝试其他可能的属性名
            for attr_name in ['text_encoder', 'encoder', 'model']:
                if hasattr(clip_model, attr_name):
                    text_model = getattr(clip_model, attr_name)
                    print(f"使用替代属性: {attr_name}")
                    break
            else:
                print(f"找不到合适的文本模型属性")
                return False
                
        # 检查是否有text_projection属性
        has_text_projection = hasattr(clip_model, 'text_projection')
        print(f"是否有text_projection属性: {has_text_projection}")
        
        # 如果没有text_projection，创建一个
        if not has_text_projection:
            print(f"CLIP模型缺少text_projection属性，尝试创建...")
            
            # 确定创建参数需要的形状
            # 通常text_projection.weight的形状是[hidden_size, projection_dim]
            hidden_size = None
            projection_dim = None
            
            # 尝试从text_model获取hidden_size
            if hasattr(text_model, 'config') and hasattr(text_model.config, 'hidden_size'):
                hidden_size = text_model.config.hidden_size
                print(f"从text_model.config获取hidden_size: {hidden_size}")
            elif hasattr(text_model, 'hidden_size'):
                hidden_size = text_model.hidden_size
                print(f"从text_model直接获取hidden_size: {hidden_size}")
            elif hasattr(clip_model, 'config') and hasattr(clip_model.config, 'text_config') and hasattr(clip_model.config.text_config, 'hidden_size'):
                hidden_size = clip_model.config.text_config.hidden_size
                print(f"从clip_model.config.text_config获取hidden_size: {hidden_size}")
            else:
                # 尝试从最后一层的权重形状推断
                if hasattr(text_model, 'encoder') and hasattr(text_model.encoder, 'layers'):
                    last_layer = text_model.encoder.layers[-1]
                    if hasattr(last_layer, 'self_attn') and hasattr(last_layer.self_attn, 'out_proj') and hasattr(last_layer.self_attn.out_proj, 'weight'):
                        hidden_size = last_layer.self_attn.out_proj.weight.shape[0]
                        print(f"从最后一层注意力权重推断hidden_size: {hidden_size}")
                
            # 尝试从视觉投影或配置获取projection_dim
            if hasattr(clip_model, 'visual_projection') and hasattr(clip_model.visual_projection, 'weight'):
                projection_dim = clip_model.visual_projection.weight.shape[0]
                print(f"从visual_projection权重推断projection_dim: {projection_dim}")
            elif hasattr(clip_model, 'config') and hasattr(clip_model.config, 'projection_dim'):
                projection_dim = clip_model.config.projection_dim
                print(f"从config获取projection_dim: {projection_dim}")
            elif hasattr(clip_model, 'embed_dim'):
                projection_dim = clip_model.embed_dim
                print(f"从embed_dim获取projection_dim: {projection_dim}")
            
            # 如果找不到确切维度，使用标准CLIP的默认值
            if hidden_size is None:
                hidden_size = 768  # 标准CLIP默认值
                print(f"使用默认hidden_size: {hidden_size}")
            
            if projection_dim is None:
                projection_dim = 768  # 标准CLIP默认值
                print(f"使用默认projection_dim: {projection_dim}")
            
            # 创建text_projection权重并添加到模型
            device = next(clip_model.parameters()).device
            dtype = next(clip_model.parameters()).dtype
            
            print(f"创建text_projection权重: [{hidden_size}, {projection_dim}], 设备: {device}, 类型: {dtype}")
            # 使用Xavier初始化以保持良好的梯度流
            text_projection = torch.nn.Linear(hidden_size, projection_dim, bias=False)
            torch.nn.init.xavier_uniform_(text_projection.weight)
            text_projection = text_projection.to(device, dtype=dtype)
            
            # 将text_projection添加到模型
            clip_model.text_projection = text_projection
            print(f"已成功添加text_projection到CLIP模型")
            
            # 验证添加是否成功
            if hasattr(clip_model, 'text_projection'):
                print(f"验证成功: text_projection已添加到模型")
                return True
            else:
                print(f"验证失败: text_projection未添加到模型")
                return False
        else:
            print(f"CLIP模型已有text_projection属性，无需修复")
            return True
    except Exception as e:
        print(f"修复CLIP模型时出错: {str(e)}")
        traceback.print_exc()
        return False

def patch_clip_text_encoder_forward(clip_model: Any) -> bool:
    """
    为CLIP的text_encoder的forward方法添加补丁，确保text_projection被正确应用
    
    Args:
        clip_model: CLIP模型对象
        
    Returns:
        bool: 是否成功应用补丁
    """
    try:
        import types
        
        if clip_model is None:
            print("CLIP模型为空，无法应用补丁")
            return False
            
        # 确定text_encoder
        text_encoder = None
        if hasattr(clip_model, 'text_model'):
            text_encoder = clip_model.text_model
        elif hasattr(clip_model, 'transformer'):
            text_encoder = clip_model.transformer
        elif hasattr(clip_model, 'text_encoder'):
            text_encoder = clip_model.text_encoder
        else:
            print("找不到CLIP文本编码器，无法应用补丁")
            return False
            
        # 确保模型有text_projection
        if not hasattr(clip_model, 'text_projection'):
            print("CLIP模型缺少text_projection，无法应用补丁")
            return False
            
        # 保存原始forward方法
        original_forward = text_encoder.forward
        
        # 定义新的forward方法
        def patched_forward(self, *args, **kwargs):
            # 调用原始forward方法获取输出
            outputs = original_forward(*args, **kwargs)
            
            # 检查输出格式
            if isinstance(outputs, tuple) and len(outputs) > 0:
                # 典型的transformers模型输出格式
                last_hidden_state = None
                
                # 尝试从输出中获取last_hidden_state
                if hasattr(outputs, 'last_hidden_state'):
                    last_hidden_state = outputs.last_hidden_state
                elif isinstance(outputs[0], torch.Tensor):
                    last_hidden_state = outputs[0]  # 通常第一个元素是last_hidden_state
                
                if last_hidden_state is not None:
                    # 获取pooled_output (通常是CLS token的输出)
                    pooled_output = None
                    
                    # 尝试从输出中获取pooled_output
                    if hasattr(outputs, 'pooler_output'):
                        pooled_output = outputs.pooler_output
                    elif len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
                        pooled_output = outputs[1]  # 通常第二个元素是pooled_output
                    elif hasattr(self, 'pooler') and hasattr(self.pooler, 'forward'):
                        # 如果输出中没有，尝试通过pooler获取
                        pooled_output = self.pooler(last_hidden_state)
                    else:
                        # 使用第一个token (CLS token) 作为pooled_output
                        pooled_output = last_hidden_state[:, 0]
                    
                    # 应用text_projection
                    text_embeds = clip_model.text_projection(pooled_output)
                    
                    # 标准化向量
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    
                    # 修改outputs以包含投影后的嵌入
                    if hasattr(outputs, '_asdict'):  # namedtuple
                        outputs_dict = outputs._asdict()
                        outputs_dict['text_embeds'] = text_embeds
                        # 重新创建namedtuple
                        from collections import namedtuple
                        OutputType = namedtuple(type(outputs).__name__, outputs_dict.keys())
                        outputs = OutputType(**outputs_dict)
                    elif isinstance(outputs, tuple):
                        # 为tuple添加新元素
                        outputs = outputs + (text_embeds,)
            
            return outputs
        
        # 应用补丁
        text_encoder.forward = types.MethodType(patched_forward, text_encoder)
        print("已成功应用CLIP文本编码器forward方法补丁")
        return True
    except Exception as e:
        print(f"应用CLIP补丁时出错: {str(e)}")
        traceback.print_exc()
        return False
