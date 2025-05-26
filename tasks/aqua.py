from typing import List
import re
import os
import json
import requests

from llm_client import LLMClient
from tasks.base import Task
from utils import Example


class AQUA(Task):
    def __init__(self, llm: LLMClient):
        super().__init__("aqua", llm)

    def load_data(self) -> List[Example]:
        data = []
        
        print("正在从GitHub加载AQuA数据集...")
        
        try:
            # 直接从GitHub仓库获取AQuA测试数据集
            url = "https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json"
            response = requests.get(url)
            
            if response.status_code != 200:
                raise Exception(f"GitHub请求失败，状态码: {response.status_code}")
                
            # 获取原始文本
            raw_text = response.text
            
            # 检查文件是JSON数组还是JSONL格式（每行一个JSON对象）
            # 通过检查第一个字符判断
            if raw_text.strip().startswith('['):
                # 标准JSON数组格式
                try:
                    dataset = json.loads(raw_text)
                    print("成功解析AQuA数据集 (JSON数组格式)")
                except Exception as e:
                    raise Exception(f"JSON数组解析失败: {str(e)}")
            else:
                # 可能是JSONL格式（每行一个JSON对象）
                print("检测到可能是JSONL格式，按行解析...")
                dataset = []
                lines = raw_text.strip().split('\n')
                
                for i, line in enumerate(lines):
                    if line.strip():  # 跳过空行
                        try:
                            item = json.loads(line.strip())
                            dataset.append(item)
                        except Exception as e:
                            print(f"警告: 跳过无效的JSON行 {i+1}: {line[:50]}...")
                
                print(f"成功从JSONL格式解析 {len(dataset)} 条记录")
                
            print(f"AQuA数据集样本总数: {len(dataset)}")
            
            # 打印样本示例以便调试
            if len(dataset) > 0:
                print(f"数据样本结构: {list(dataset[0].keys())}")
            
            # 处理数据
            for i, example in enumerate(dataset):
                try:
                    # 获取问题文本和选项
                    question_text = example.get("question", "")
                    options = example.get("options", [])
                    correct_option = example.get("correct", "")
                    
                    # 确认所有必需字段都存在
                    if not question_text or not options or not correct_option:
                        print(f"警告: 第 {i} 个样本缺少必要字段，跳过")
                        continue
                    
                    # 将选项添加到问题中
                    options_text = "\nOptions:\n"
                    for j, option_text in enumerate(options):
                        option_letter = chr(65 + j)  # A, B, C, D, E
                        options_text += f"{option_letter}. {option_text}\n"
                    
                    # 组合问题和选项
                    formatted_question = question_text + options_text
                    
                    # 转换为Example格式
                    formatted_example = {
                        "question": formatted_question,
                        "answer": correct_option
                    }
                    data.append(Example.model_validate(formatted_example))
                except Exception as e:
                    print(f"处理第 {i} 个样本时出错: {str(e)}")
                    continue
                    
            if len(data) == 0:
                raise Exception("没有成功处理任何样本")
                
        except Exception as e:
            print(f"AQuA数据集加载失败: {str(e)}")
            print("尝试从缓存加载...")
            
            try:
                # 尝试从本地缓存加载(如果之前下载过)
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "aqua")
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, "aqua_test.json")
                
                if os.path.exists(cache_file):
                    print(f"从缓存文件加载: {cache_file}")
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        dataset = json.load(f)
                    print(f"从缓存加载AQuA数据集，样本数: {len(dataset)}")
                    
                    for i, example in enumerate(dataset):
                        try:
                            # 获取问题文本和选项
                            question_text = example.get("question", "")
                            options = example.get("options", [])
                            correct_option = example.get("correct", "")
                            
                            # 将选项添加到问题中
                            options_text = "\nOptions:\n"
                            for j, option_text in enumerate(options):
                                option_letter = chr(65 + j)  # A, B, C, D, E
                                options_text += f"{option_letter}. {option_text}\n"
                            
                            # 组合问题和选项
                            formatted_question = question_text + options_text
                            
                            # 转换为Example格式
                            formatted_example = {
                                "question": formatted_question,
                                "answer": correct_option
                            }
                            data.append(Example.model_validate(formatted_example))
                        except Exception as e:
                            print(f"处理第 {i} 个缓存样本时出错: {str(e)}")
                            continue
                else:
                    raise Exception("本地缓存文件不存在")
            except Exception as e:
                print(f"从缓存加载失败: {str(e)}")
                raise Exception("无法加载AQuA数据集")
        
        # 缓存数据集以便离线使用
        if len(data) > 0:
            try:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "aqua")
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, "aqua_test.json")
                
                # 将数据集保存为一个有效的JSON数组
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"已将AQuA数据集缓存到 {cache_file}")
            except Exception as e:
                print(f"缓存数据集时出错: {str(e)}")
        
        print(f"成功处理AQuA样本: {len(data)}个")
        return data

    def extract_answer(self, raw_response: str) -> str:
        """提取选择题答案 (A/B/C/D/E)"""
        # 保存原始回答用于备用提取
        original_response = raw_response
        
        # 处理####分隔符，如果存在
        if "####" in raw_response:
            parts = raw_response.split("####")
            
            # 先尝试从####后面提取
            if len(parts) > 1 and parts[-1].strip():
                post_delimiter = parts[-1].strip()
                result = self._extract_option_from_text(post_delimiter)
                if result and result in ['A', 'B', 'C', 'D', 'E']:
                    return result
            
            # 如果####后面没有有效答案，尝试从####前面提取
            pre_delimiter = parts[0].strip()
            if pre_delimiter:
                result = self._extract_option_from_text(pre_delimiter)
                if result and result in ['A', 'B', 'C', 'D', 'E']:
                    return result
        
        # 如果没有分隔符或分隔符处理失败，处理整个回答
        return self._extract_option_from_text(original_response)
    
    def _extract_option_from_text(self, text: str) -> str:
        """从文本中提取选项字母"""
        # 规范化答案文本
        text = text.upper().strip()
        
        # 尝试多种模式提取字母选项
        patterns = [
            r'ANSWER[:\s]*([A-E])',        # "ANSWER: A" 格式
            r'FINAL ANSWER[:\s]*([A-E])',  # "FINAL ANSWER: A" 格式
            r'选择\s*([A-E])',              # "选择 A" 格式
            r'OPTION\s*([A-E])',           # "OPTION A" 格式
            r'选项\s*([A-E])',              # "选项 A" 格式
            r'([A-E])[.)\s]*$',            # 末尾的 "A." 格式
            r'→\s*([A-E])',                # 箭头指向格式
            r'\b([A-E])\b',                # 独立的 "A" 格式
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # 如果无法提取字母，尝试提取最后一个大写字母
        matches = re.findall(r'[A-E]', text)
        if matches:
            return matches[-1]
            
        print(f"警告: 无法从回答中提取选项: {text}")
        return text

    def equal(self, predicted_answer: str, expected_answer: str) -> bool:
        # 规范化答案格式
        predicted_clean = predicted_answer.strip().upper()
        expected_clean = expected_answer.strip().upper()
        
        # 直接比较第一个字母
        if len(predicted_clean) > 0 and len(expected_clean) > 0:
            return predicted_clean[0] == expected_clean[0]
        
        return predicted_clean == expected_clean