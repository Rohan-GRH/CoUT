import json
import os
import re
from typing import List

from pydantic import BaseModel

from utils import Example
from tasks.tale_base import TaleTask


class GSM8KTale(TaleTask):
    def __init__(self, llm):
        super().__init__("gsm8k", llm)  # 保持任务名称为gsm8k以便使用正确的配置

    def load_data(self) -> List[Example]:
        # 数据路径
        data_path = "./data/gsm8k/test.jsonl"
        
        if not os.path.exists(data_path):
            # 如果文件不存在，尝试从Hugging Face下载
            try:
                from datasets import load_dataset

                dataset = load_dataset("gsm8k", "main")
                # 保存到本地路径
                if not os.path.exists("./data/gsm8k"):
                    os.makedirs("./data/gsm8k")
                
                with open(data_path, "w") as f:
                    for example in dataset["test"]:
                        f.write(json.dumps(example) + "\n")
                print(f"已从Hugging Face下载GSM8K数据集并保存到{data_path}")
            except Exception as e:
                raise Exception(f"无法下载GSM8K数据集: {e}")

        # 读取数据
        examples = []
        with open(data_path, "r") as f:
            for line in f:
                example = json.loads(line)
                question = example["question"]
                
                # 使用正则表达式提取答案
                answer_match = re.search(r"####\s*(.*?)$", example["answer"], re.DOTALL)
                if answer_match:
                    answer = answer_match.group(1).strip()
                    examples.append(Example(question=question, answer=answer))
                else:
                    print(f"无法从以下文本中提取答案: {example['answer']}")
        
        return examples

    def extract_answer(self, response: str) -> str:
        """从模型回答中提取纯数字答案，处理各种格式的回答"""
        import re
        
        # 如果有####分隔符，尝试从后面提取答案
        if "####" in response:
            parts = response.split("####")
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
        raw_response = response.strip()
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
        # 首先检查是否都为空
        if not predicted_answer or not expected_answer:
            return False
        
        # 去除空格和逗号进行基本比较
        predicted_clean = predicted_answer.strip().replace(",", "")
        expected_clean = expected_answer.strip().replace(",", "")
        
        if predicted_clean == expected_clean:
            return True
        
        # 尝试提取数字进行比较
        try:
            # 清理非数字字符（保留负号和小数点）
            predicted_num = re.sub(r'[^-\d.]+', '', predicted_clean)
            expected_num = re.sub(r'[^-\d.]+', '', expected_clean)
            
            # 转换为浮点数比较
            predicted_float = float(predicted_num)
            expected_float = float(expected_num)
            
            # 对于小数，允许一定的误差
            return abs(predicted_float - expected_float) < 1e-6
        except (ValueError, TypeError):
            # 如果无法转换为浮点数，回退到字符串比较
            return predicted_clean == expected_clean
