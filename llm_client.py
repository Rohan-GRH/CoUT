import os

from anthropic import Anthropic
from openai import OpenAI


class LLMClient:
    def __init__(self, base_url: str = None, api_key: str = None):
        
        self.api_key = api_key
        
        
        self.is_deepinfra = base_url and "deepinfra.com" in base_url
        
        
        if self.is_deepinfra:
            if base_url.endswith("/v1/openai/chat/completions"):
                # 移除chat/completions部分，因为这会在OpenAI客户端中自动添加
                self.base_url = base_url.replace("/chat/completions", "")
            elif not base_url.endswith("/v1/openai") and not base_url.endswith("/v1/openai/"):
                if not base_url.endswith("/"):
                    self.base_url = f"{base_url}/v1/openai"
                else:
                    self.base_url = f"{base_url}v1/openai"
            else:
                self.base_url = base_url
        else:
            self.base_url = base_url
        
        # 创建OpenAI客户端
        # 注意：对于DeepInfra，我们不在这里设置Bearer前缀，而是在请求头中设置
        self.openai_client = OpenAI(base_url=self.base_url, api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY")
        self.anthropic_client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def request(
        self,
        payload: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        if model.startswith("claude"):
            message = self.anthropic_client.messages.create(
                messages=[{"role": "user", "content": payload}],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = message.content[0].text
            token_count = message.usage.output_tokens
        else:
            # 处理DeepInfra API的特殊情况
            if self.is_deepinfra:
                # 为DeepInfra API设置正确的请求头
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    # 确保使用Bearer格式的认证头
                    auth_value = self.api_key
                    if not auth_value.startswith("Bearer "):
                        auth_value = f"Bearer {auth_value}"
                    headers["Authorization"] = auth_value
                
                # 对于DeepInfra，需要使用正确的模型名称格式
                # 如果模型名称是MODEL_MAPPING中的简写，需要转换为完整名称
                if model == "deepseek-r1":
                    model_name = "deepseek-ai/DeepSeek-R1"
                elif model == "qwen-qwq-32b":
                    model_name = "Qwen/QwQ-32B"
                else:
                    model_name = model
                
                # 创建请求
                # 对于新模型（如o3-mini），使用max_completion_tokens而不是max_tokens
                # 并且移除不支持的temperature参数
                if "o3-mini" in model:
                    completion = self.openai_client.chat.completions.create(
                        messages=[{"role": "user", "content": payload}],
                        model=model_name,
                        max_completion_tokens=max_tokens,
                        extra_headers=headers
                    )
                else:
                    completion = self.openai_client.chat.completions.create(
                        messages=[{"role": "user", "content": payload}],
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_headers=headers
                    )
            else:
                # 标准OpenAI API调用
                # 对于新模型（如o3-mini），使用max_completion_tokens而不是max_tokens
                # 并且移除不支持的temperature参数
                if "o3-mini" in model:
                    completion = self.openai_client.chat.completions.create(
                        messages=[{"role": "user", "content": payload}],
                        model=model,
                        max_completion_tokens=max_tokens,
                    )
                else:
                    completion = self.openai_client.chat.completions.create(
                        messages=[{"role": "user", "content": payload}],
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
            
            response = completion.choices[0].message.content
            token_count = completion.usage.completion_tokens
        return response, token_count


if __name__ == "__main__":
    llm = LLMClient()
    response, count = llm.request("hello", "claude-3-7-sonnet-latest")
    print(response, count)
