from typing import List
import os
import json

from datasets import load_dataset

from llm_client import LLMClient
from tasks.base import Task
from utils import Example


class MathQA(Task):
    def __init__(self, llm: LLMClient):
        super().__init__("mathqa", llm)

    def load_data(self) -> List[Example]:
        data = []
        
        # 尝试加载数据集，添加trust_remote_code=True参数
        try:
            dataset = load_dataset("math_qa", split="test", trust_remote_code=True)
            print(f"成功加载MathQA数据集，样本数: {len(dataset)}")
        except Exception as e:
            print(f"加载MathQA数据集失败: {e}")
            try:
                # 尝试替代方案
                dataset = load_dataset("allenai/math_qa", split="test", trust_remote_code=True)
                print(f"成功加载allenai/math_qa数据集，样本数: {len(dataset)}")
            except Exception as e:
                print(f"所有MathQA数据集尝试都失败: {e}")
                # 使用内置样本
                print("使用内置备选数据")
                return self._get_fallback_examples()
        
        # 处理数据
        try:
            for example in dataset:
                # 转换为Example格式
                formatted_example = {
                    "question": example.get("Problem", "") + " " + example.get("options", ""),
                    "answer": example.get("correct", "")
                }
                data.append(Example.model_validate(formatted_example))
                
                # 限制样本数量
                if len(data) >= 5000:
                    break
        except Exception as e:
            print(f"处理MathQA样本时出错: {e}")
            return self._get_fallback_examples()
            
        return data

    def _get_fallback_examples(self) -> List[Example]:
        """当无法从Hugging Face加载数据集时，提供一些内置的示例"""
        examples = [
            {
                "question": "A watch loses 5 minutes every hour. If it is set correctly at 8 AM, what time will it show at 8 PM the same day? Options: a) 7:00 PM b) 6:40 PM c) A different time d) 7:40 PM e) 7:20 PM",
                "answer": "a"
            },
            {
                "question": "What is the greatest common divisor of 84 and 90? Options: a) 6 b) 42 c) 2 d) 3 e) 1",
                "answer": "a"
            },
            {
                "question": "If log(x) = 3 and log(y) = 4, then log(x^2 * y) = ? Options: a) 7 b) 10 c) 24 d) 12 e) 5",
                "answer": "b"
            },
            {
                "question": "Mary is twice as old as Jane was when Mary was as old as Jane is now. If Mary is 24 years old, how old is Jane? Options: a) 12 years b) 16 years c) 20 years d) 18 years e) 24 years",
                "answer": "d"
            },
            {
                "question": "If a box contains 6 red marbles, 4 blue marbles, and 8 green marbles, what is the probability of drawing a blue marble? Options: a) 4/18 b) 2/9 c) 1/3 d) 4/6 e) 2/3",
                "answer": "b"
            },
            {
                "question": "A car travels 50 miles at an average speed of 50 mph. How long does the trip take? Options: a) 2 hours b) 1 hour c) 0.5 hours d) 2.5 hours e) 1.5 hours",
                "answer": "b"
            },
            {
                "question": "Find the value of x in the equation 3x - 7 = 14. Options: a) 3 b) 7 c) 5 d) 9 e) 6",
                "answer": "b"
            }
        ]
        
        return [Example.model_validate(example) for example in examples]

    def extract_answer(self, raw_response: str) -> str:
        """从模型回答中提取选项 (a/b/c/d/e)"""
        import re
        
        # 保存原始回答用于备用提取
        original_response = raw_response
        
        # 处理分隔符
        if "####" in raw_response:
            # 先尝试从####后面提取
            post_delimiter = raw_response.split("####")[-1].strip()
            
            # 如果####后面有内容，尝试提取
            if post_delimiter:
                result = self._extract_option_from_text(post_delimiter)
                if result and result.lower() in ['a', 'b', 'c', 'd', 'e']:
                    return result
            
            # 如果####后面没有有效答案，尝试从####前面提取
            pre_delimiter = raw_response.split("####")[0].strip()
            if pre_delimiter:
                result = self._extract_option_from_text(pre_delimiter)
                if result and result.lower() in ['a', 'b', 'c', 'd', 'e']:
                    return result
        
        # 如果没有分隔符或分隔符处理失败，处理整个回答
        return self._extract_option_from_text(original_response)
    
    def _extract_option_from_text(self, text: str) -> str:
        """从文本中提取选项字母"""
        import re
        text = text.lower().strip()
        
        # 尝试多种模式提取字母选项
        # 1. 查找"答案是X"或"answer is X"格式
        patterns = [
            r'答案是\s*([a-e])',           # 中文格式
            r'answer\s*(?:is|:|=)\s*([a-e])',  # 英文格式
            r'final\s*answer\s*(?:is|:|=)\s*([a-e])',  # Final answer格式
            r'选择\s*([a-e])',             # 选择X格式
            r'option\s*([a-e])',          # Option X格式
            r'选项\s*([a-e])',             # 选项X格式
            r'([a-e])[.)\s:]*$',          # 末尾的X格式
            r'→\s*([a-e])',               # 箭头指向格式
            r'\b([a-e])\b'                # 独立的a/b/c/d/e格式
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # 如果上述模式都没找到，尝试提取最后一个选项字母
        matches = re.findall(r'[a-e]', text)
        if matches:
            return matches[-1]
            
        return text

    def equal(self, predicted_answer: str, expected_answer: str) -> bool:
        # 标准化答案
        predicted_answer = predicted_answer.strip().lower()
        expected_answer = expected_answer.strip().lower()
        
        # 直接比较标准化后的答案
        return predicted_answer == expected_answer