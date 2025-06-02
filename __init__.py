# 设置protobuf环境变量以解决protobuf兼容性问题
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import folder_paths

# 导入模块化节点设计
from .nodes import SigLIP_Loader, DINOv2_Loader, IPAdapter_Loader, InstantCharacter, load_instantcharacter_paths, FluxInstantCharacter

NODE_CLASS_MAPPINGS = {
    "SigLIP_Loader": SigLIP_Loader,
    "DINOv2_Loader": DINOv2_Loader,
    "IPAdapter_Loader": IPAdapter_Loader,
    "InstantCharacter": InstantCharacter,
    "FluxInstantCharacter": FluxInstantCharacter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SigLIP_Loader": "SigLIP模型加载器",
    "DINOv2_Loader": "DINOv2模型加载器",
    "IPAdapter_Loader": "IP-Adapter模型加载器",
    "InstantCharacter": "InstantCharacter",
    "FluxInstantCharacter": "InstantCharacter(FLUX)"
}
