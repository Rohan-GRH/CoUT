from typing import List
import re
import os
import json
import requests

from llm_client import LLMClient
from tasks.base import Task
from utils import Example


class SVAMP(Task):
    def __init__(self, llm: LLMClient):
        super().__init__("svamp", llm)

    def load_data(self) -> List[Example]:
        data = []
        
        print("正在从GitHub加载SVAMP数据集...")
        
        try:
            # 直接从SVAMP的GitHub仓库加载数据集
            url = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"
            response = requests.get(url)
            
            if response.status_code != 200:
                raise Exception(f"GitHub请求失败，状态码: {response.status_code}")
                
            # 加载JSON数据
            dataset = json.loads(response.text)
            print(f"成功加载SVAMP数据集，样本数: {len(dataset)}")
            
            # 打印样本示例以便调试
            if len(dataset) > 0:
                print(f"数据样本示例: {dataset[0].keys()}")
            
            # 处理数据
            for i, example in enumerate(dataset):
                try:
                    # 组合Body和Question字段
                    body = example.get("Body", "")
                    question = example.get("Question", "")
                    answer = str(example.get("Answer", ""))
                    
                    # 完整问题文本
                    full_question = f"{body} {question}"
                    
                    # 转换为Example格式
                    formatted_example = {
                        "question": full_question,
                        "answer": answer
                    }
                    data.append(Example.model_validate(formatted_example))
                except Exception as e:
                    print(f"处理第 {i} 个样本时出错: {str(e)}")
                    continue
                    
            if len(data) == 0:
                raise Exception("没有成功处理任何样本")
                
        except Exception as e:
            print(f"SVAMP数据集加载失败: {str(e)}")
            print("尝试从缓存加载...")
            
            try:
                # 尝试从本地缓存加载(如果之前下载过)
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "svamp")
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, "svamp.json")
                
                if os.path.exists(cache_file):
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        dataset = json.load(f)
                    print(f"从缓存加载SVAMP数据集，样本数: {len(dataset)}")
                    
                    for i, example in enumerate(dataset):
                        try:
                            # 组合Body和Question字段
                            body = example.get("Body", "")
                            question = example.get("Question", "")
                            answer = str(example.get("Answer", ""))
                            
                            # 完整问题文本
                            full_question = f"{body} {question}"
                            
                            # 转换为Example格式
                            formatted_example = {
                                "question": full_question,
                                "answer": answer
                            }
                            data.append(Example.model_validate(formatted_example))
                        except Exception as e:
                            print(f"处理第 {i} 个缓存样本时出错: {str(e)}")
                            continue
                else:
                    raise Exception("本地缓存文件不存在")
            except Exception as e:
                print(f"从缓存加载失败: {str(e)}")
                raise Exception("无法加载SVAMP数据集")
        
        # 缓存数据集以便离线使用
        if len(data) > 0:
            try:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "svamp")
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, "svamp.json")
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"已将SVAMP数据集缓存到 {cache_file}")
            except Exception as e:
                print(f"缓存数据集时出错: {str(e)}")
        
        print(f"成功处理SVAMP样本: {len(data)}个")
        return data

    def extract_answer(self, raw_response: str) -> float:
        """从模型回答中提取数值答案"""
        # 处理分隔符
        if "####" in raw_response:
            # 尝试从####后面提取
            post_delimiter = raw_response.split("####")[-1].strip()
            
            # 如果####后面有内容，尝试提取数值
            if post_delimiter:
                result = self._extract_numeric_answer(post_delimiter)
                if result is not None:
                    return result
            
            # 如果####后面没有有效答案，尝试从####前面提取
            pre_delimiter = raw_response.split("####")[0].strip()
            if pre_delimiter:
                result = self._extract_numeric_answer(pre_delimiter)
                if result is not None:
                    return result
        
        # 如果没有分隔符或分隔符处理失败，处理整个回答
        return self._extract_numeric_answer(raw_response)
    
    def _extract_numeric_answer(self, text: str) -> float:
        """提取文本中的数值答案"""
        # 正则表达式模式匹配答案部分
        answer_patterns = [
            r'answer is ([-+]?\d*\.?\d+)', 
            r'answer: ([-+]?\d*\.?\d+)',
            r'answer = ([-+]?\d*\.?\d+)',
            r'= ([-+]?\d*\.?\d+)$',
            r'答案是 ([-+]?\d*\.?\d+)',
            r'答案为 ([-+]?\d*\.?\d+)',
            r'答案：([-+]?\d*\.?\d+)',
            r'答案: ([-+]?\d*\.?\d+)',
            r'最终答案是 ([-+]?\d*\.?\d+)',
            r'最终答案：([-+]?\d*\.?\d+)',
            r'the answer is ([-+]?\d*\.?\d+)',
            r'final answer is ([-+]?\d*\.?\d+)',
            r'final answer: ([-+]?\d*\.?\d+)',
        ]
        
        # 尝试每个模式
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # 如果以上模式都未匹配，尝试提取所有数字并使用最后一个
        numbers = re.findall(r'([-+]?\d*\.?\d+)', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        # 如果还是无法提取，返回原始文本
        print(f"警告：无法从响应中提取数值: '{text}'")
        return 0.0  # 返回0作为默认值

    def equal(self, predicted_answer: float, expected_answer: float) -> bool:
        """检查预测答案与期望答案是否相等，允许小的浮点误差"""
        try:
            # 尝试转换为浮点数进行比较
            pred_float = float(predicted_answer)
            expected_float = float(expected_answer)
            
            # 对于整数答案，进行精确比较
            if expected_float.is_integer():
                return int(pred_float) == int(expected_float)
            
            # 对于浮点数答案，允许小的误差
            # 使用相对误差，对于大数值更为合理
            if abs(expected_float) > 1e-10:  # 非零值使用相对误差
                relative_error = abs((pred_float - expected_float) / expected_float)
                return relative_error < 0.001  # 允许0.1%的误差
            else:  # 对于接近零的值使用绝对误差
                return abs(pred_float - expected_float) < 1e-10
            
        except (ValueError, TypeError):
            # 如果无法转换为浮点数，直接比较字符串表示
            return str(predicted_answer).strip() == str(expected_answer).strip()