from typing import List

from datasets import load_dataset

from llm_client import LLMClient
from tasks.base import Task
from utils import Example, extract_number_from_string


class GSM8K(Task):
    def __init__(self, llm: LLMClient):
        super().__init__("gsm8k", llm)

    def load_data(self) -> List[Example]:
        data = []
        for example in load_dataset("openai/gsm8k", "main", split="test"):
            data.append(Example.model_validate(example))
        return data

    def extract_answer(self, raw_response: str) -> str:
        """从模型回答中提取纯数字答案，处理各种格式的回答"""
        import re
        
        # 如果有####分隔符，尝试从后面提取答案
        if "####" in raw_response:
            parts = raw_response.split("####")
            # 如果####后面有内容，优先使用
            if len(parts) > 1 and parts[1].strip():
                after_part = parts[1].strip()
                # 清理非数字字符，但保留负号和小数点
                after_part = re.sub(r'[^-\d.]+', '', after_part)
                numbers = re.findall(r'-?\d+\.?\d*', after_part)
                if numbers:
                    # 清理末尾的句点（如果不是小数点的一部分）
                    answer = numbers[0]
                    if answer.endswith('.') and not re.match(r'-?\d+\.\d+', answer):
                        answer = answer[:-1]
                    return answer
            
            # 如果####后面没有内容或没有找到数字，从前面部分提取最后一个数字
            before_part = parts[0].strip()
            # 清理非数字字符，但保留负号和小数点
            before_part = re.sub(r'[^-\d.]+', '', before_part)
            # 查找前面部分中的所有数字
            numbers = re.findall(r'-?\d+\.?\d*', before_part)
            if numbers:
                # 返回最后一个数字，通常是计算的最终结果
                answer = numbers[-1]
                if answer.endswith('.') and not re.match(r'-?\d+\.\d+', answer):
                    answer = answer[:-1]
                return answer
        
        # 如果没有####分隔符，处理整个回答
        raw_response = raw_response.strip()
        # 清理非数字字符，但保留负号和小数点
        cleaned_response = re.sub(r'[^-\d.]+', '', raw_response)
        
        # 查找特定模式，如"total=X"或"sum=X"
        total_match = re.search(r'(?:total|sum|answer)\s*[=:]\s*(-?\d+\.?\d*)', raw_response.lower())
        if total_match:
            answer = total_match.group(1)
            if answer.endswith('.') and not re.match(r'-?\d+\.\d+', answer):
                answer = answer[:-1]
            return answer
        
        # 使用正则表达式提取所有数字（整数或小数）
        numbers = re.findall(r'-?\d+\.?\d*', cleaned_response)
        if numbers:
            # 如果有多个数字，返回最后一个，因为它通常是最终答案
            answer = numbers[-1]
            if answer.endswith('.') and not re.match(r'-?\d+\.\d+', answer):
                answer = answer[:-1]
            return answer
        
        # 如果没有找到任何数字，返回清理后的响应
        return cleaned_response

    def equal(self, predicted_answer: str, expected_answer: str) -> bool:
        if predicted_answer == expected_answer:
            return True
        predicted_answer = str(extract_number_from_string(predicted_answer))
        if predicted_answer == expected_answer:
            return True
        try:
            if float(predicted_answer) == float(expected_answer):
                return True
        except Exception:
            return False
        return False
