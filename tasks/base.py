import time
from abc import ABC, abstractmethod
from typing import List, Literal
import random

from tqdm import tqdm

from llm_client import LLMClient
from utils import Config, Example, compose_request, load_config


class Task(ABC):
    def __init__(self, name: str, llm: LLMClient):
        self.name = name
        self.llm = llm
        self.token_count_tracker = []
        self.latency_tracker = []
        self.detailed_results = []
        self.empty_response_count = 0  # 添加空响应计数器

    @abstractmethod
    def load_data(self) -> List[Example]:
        pass

    @abstractmethod
    def extract_answer(self, raw_response: str) -> any:
        pass

    @abstractmethod
    def equal(self, predicted_answer: any, expected_answer: any) -> bool:
        pass

    def evaluate_example(self, model: str, config: Config, shot: int, example: Example) -> bool:
        # prepare payload
        payload = compose_request(config, shot, example.question)

        # 添加重试机制和空响应统计
        max_retries = 3  # 最大重试次数
        retry_count = 0
        empty_response = False
        
        while retry_count < max_retries:
            # run inference
            start_time = time.time()
            response, token_count = self.llm.request(payload, model)
            end_time = time.time()
            
            # 检查响应是否为空
            predicted_answer = self.extract_answer(response)
            if predicted_answer and str(predicted_answer).strip():
                # 响应不为空，跳出循环
                empty_response = False
                break
            
            # 响应为空，进行重试
            print(f"收到空响应，正在重试 ({retry_count + 1}/{max_retries})...")
            retry_count += 1
            empty_response = True
            
            # 如果已达到最大重试次数，使用最后一次的响应
            if retry_count >= max_retries:
                print(f"达到最大重试次数，使用最后一次响应")
                break
        
        # 如果不是空响应，记录token和延迟
        if not empty_response:
            self.token_count_tracker.append(token_count)
            self.latency_tracker.append(end_time - start_time)
            
            # check result
            expected_answer = self.extract_answer(example.answer)
            equal = self.equal(predicted_answer, expected_answer)
            
            # 记录详细结果
            self.detailed_results.append((example.question, expected_answer, predicted_answer, response, equal))
            
            if not equal:
                print(f"Example: {example.question}")
                print(f"Expected: {expected_answer}, Predicted: {predicted_answer}")
                print(f"Full response: {response}")
            return equal
        else:
            # 空响应情况，记录但不计入评估结果
            print(f"警告：问题 '{example.question}' 在多次尝试后仍然返回空响应，不计入评估结果")
            # 返回None表示此样本不应计入总体评估
            return None

    def evaluate(self, model: str, config: Literal["baseline", "cot", "cod", "CoUT"], shot: int = None, max_samples: int = None) -> float:
        correct = 0
        # 清空之前的结果
        self.detailed_results = []
        self.token_count_tracker = []
        self.latency_tracker = []
        self.empty_response_count = 0  # 重置空响应计数器
        
        config = load_config(self.name, config)
        test_set = self.load_data()
        total_samples = len(test_set)
        
        if max_samples is not None:
            if max_samples > total_samples:
                print(f"\n警告：请求的样本数 {max_samples} 超过了可用的 {total_samples} 个样本")
                print(f"使用全部 {total_samples} 个样本")
            else:
                test_set = random.sample(test_set, max_samples)
                print(f"\n随机选择了 {max_samples} 个样本（从 {total_samples} 个样本中）")
        else:
            print(f"\n使用全部 {total_samples} 个样本")
        
        valid_samples = 0  # 有效样本数（排除空响应）
        for example in tqdm(test_set):
            result = self.evaluate_example(model, config, shot, example)
            if result is None:
                # 空响应情况，不计入评估
                self.empty_response_count += 1
                continue
            
            valid_samples += 1
            if result:
                correct += 1
        
        # 输出空响应统计
        if self.empty_response_count > 0:
            print(f"\n注意：有 {self.empty_response_count} 个样本返回空响应，已从评估结果中排除")
        
        # 如果所有样本都是空响应，返回0
        if valid_samples == 0:
            print("警告：所有样本都返回了空响应，无法计算准确率")
            return 0.0
            
        return correct / valid_samples