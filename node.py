import os
import base64
import io
import torch
from PIL import Image
from openai import OpenAI
import comfy.utils

class BailianPromptGenerator:
    def __init__(self):
        self.client = None
        self.temp_files = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "dynamicPrompts": False
                }),
                "model": ("STRING", {
                    "default": "qwen-vl-max-latest",
                    "choices": ["qwen-vl-max-latest", "qwen-vl-plus"]
                }),
                "prompt_template": ("STRING", {
                    "default": (
                        "你是一个flux文生图模型的prompt生成专家，请描述这张照片，并添加光感，光线质量，质感、高清、"
                        "相机参数（f1.2大光圈，sony 相机,50mm镜头, Raw格式）等细节,对于照片中的动漫元素请转换成"
                        "现实的真实元素进行描述。以flux的prompt进行输出，请使用英文，只需要给出prompt"
                    ),
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Bailian"

    def validate_inputs(self, api_key, model):
        if not api_key and not os.getenv('DASHSCOPE_API_KEY'):
            raise ValueError("API密钥缺失")
        if model not in ["qwen-vl-max-latest", "qwen-vl-plus"]:
            raise ValueError(f"不支持的模型: {model}")

    def tensor_to_base64(self, image_tensor):
        # 确保张量维度正确 [B, H, W, C]
        if image_tensor.dim() not in [3, 4]:
            raise ValueError(f"无效张量维度：{image_tensor.shape}")

        # 处理批次维度
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]

        # 转换为PIL图像
        image_np = image_tensor.cpu().numpy()
        image_np = (image_np * 255).astype('uint8')
        if image_np.shape[2] == 4:  # 处理RGBA格式
            image = Image.fromarray(image_np[:, :, :3], 'RGB')
        else:
            image = Image.fromarray(image_np, 'RGB')

        # 内存缓冲处理
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        return base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    def generate_prompt(self, **kwargs):
        try:
            # 输入验证
            self.validate_inputs(kwargs['api_key'], kwargs['model'])

            # 初始化客户端（延迟加载）
            if not self.client:
                self.client = OpenAI(
                    api_key=kwargs['api_key'] or os.getenv('DASHSCOPE_API_KEY'),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )

            # 转换图像张量
            base64_image = self.tensor_to_base64(kwargs['image'])

            # API请求
            completion = self.client.chat.completions.create(
                model=kwargs['model'],
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": kwargs['prompt_template']}
                    ]}
                ],
                timeout=30
            )

            # 处理响应
            if not completion.choices:
                return ("[错误] API返回空响应",)

            result = completion.choices[0].message.content
            return (result.strip(),)
        
        except Exception as e:
            # 确保始终返回有效输出
            error_msg = f"[错误] {str(e)}"
            print(f"节点执行失败: {error_msg}")
            return (error_msg,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "BailianPromptGenerator": BailianPromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BailianPromptGenerator": "百炼 Prompt 生成器 (稳定版)"
}