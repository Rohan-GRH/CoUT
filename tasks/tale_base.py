import time
import random
from typing import List, Literal, Tuple, Dict, Any
from abc import ABC

from tqdm import tqdm

from tasks.base import Task
from utils import load_config
from utils_tale_ep import tale_ep_evaluate_example


class TaleTask(Task):
    """TALE-EP任务基类，扩展了基础任务类来支持TALE-EP评估方法"""
    
    def evaluate_tale_ep(self, model: str, config_type: Literal["tale_ep"], shot: int = None, max_samples: int = None) -> Dict[str, Any]:
        """
        使用TALE-EP方法评估任务
        
        Args:
            model: 使用的模型
            config_type: 配置类型，必须是"tale_ep"
            shot: few-shot示例数量，None表示使用所有可用的few-shot示例
            max_samples: 最大评估样本数，None表示使用所有样本
            
        Returns:
            Dict[str, Any]: 包含评估结果的字典
        """
        # 清空之前的结果
        self.detailed_results = []
        self.token_count_tracker = []
        self.latency_tracker = []
        self.empty_response_count = 0
        
        # 加载配置
        config = load_config(self.name, config_type)
        test_set = self.load_data()
        total_samples = len(test_set)
        
        # 处理max_samples参数
        if max_samples is not None:
            if max_samples > total_samples:
                print(f"\n警告：请求的样本数 {max_samples} 超过了可用的 {total_samples} 个样本")
                print(f"使用全部 {total_samples} 个样本")
            else:
                test_set = random.sample(test_set, max_samples)
                print(f"\n随机选择了 {max_samples} 个样本（从 {total_samples} 个样本中）")
        else:
            print(f"\n使用全部 {total_samples} 个样本")
        
        # 统计数据
        correct = 0
        valid_samples = 0
        estimation_token_count = []  # 预估阶段使用的token数
        solution_token_count = []    # 解题阶段使用的token数
        total_token_count = []       # 总共使用的token数
        
        # 评估每个样本
        for example in tqdm(test_set):
            result = tale_ep_evaluate_example(
                task=self,
                model=model,
                config=config,
                shot=shot,
                example=example,
                llm_client=self.llm
            )
            
            if result is None:
                # 空响应情况，不计入评估
                self.empty_response_count += 1
                continue
            
            is_correct, est_tokens, sol_tokens, total_tokens = result
            valid_samples += 1
            
            if is_correct:
                correct += 1
            
            estimation_token_count.append(est_tokens)
            solution_token_count.append(sol_tokens)
            total_token_count.append(total_tokens)
            self.token_count_tracker.append(total_tokens)  # 记录总token数用于结果报告
        
        # 输出空响应统计
        if self.empty_response_count > 0:
            print(f"\n注意：有 {self.empty_response_count} 个样本返回空响应，已从评估结果中排除")
        
        # 如果所有样本都是空响应，返回0
        if valid_samples == 0:
            print("警告：所有样本都返回了空响应，无法计算准确率")
            return {
                "accuracy": 0.0,
                "estimation_token_avg": 0,
                "solution_token_avg": 0,
                "total_token_avg": 0
            }
        
        # 计算平均token使用量
        estimation_token_avg = sum(estimation_token_count) / len(estimation_token_count)
        solution_token_avg = sum(solution_token_count) / len(solution_token_count)
        total_token_avg = sum(total_token_count) / len(total_token_count)
        
        # 输出TALE-EP特有的统计信息
        print(f"\n=== TALE-EP 统计 ===")
        print(f"预估阶段平均Token: {estimation_token_avg:.2f}")
        print(f"解题阶段平均Token: {solution_token_avg:.2f}")
        print(f"Avg second_query_tokens: {solution_token_avg:.2f}")  # 添加与用户需求一致的命名
        print(f"总平均Token: {total_token_avg:.2f}")
        
        return {
            "accuracy": correct / valid_samples,
            "estimation_token_avg": estimation_token_avg,
            "solution_token_avg": solution_token_avg,
            "avg_second_query_tokens": solution_token_avg,  # 添加与用户需求一致的命名
            "total_token_avg": total_token_avg
        }
    
    def evaluate(self, model: str, config_type: Literal["baseline", "cot", "cod", "CoUT", "tale_ep"], shot: int = None, max_samples: int = None) -> float:
        """
        评估任务，根据配置类型选择评估方法
        
        Args:
            model: 使用的模型
            config_type: 配置类型
            shot: few-shot示例数量，None表示使用所有可用的few-shot示例
            max_samples: 最大评估样本数，None表示使用所有样本
            
        Returns:
            float: 准确率
        """
        if config_type == "tale_ep":
            # 使用TALE-EP评估方法
            result = self.evaluate_tale_ep(model, config_type, shot, max_samples)
            return result["accuracy"]
        else:
            # 使用原始评估方法
            return super().evaluate(model, config_type, shot, max_samples)
