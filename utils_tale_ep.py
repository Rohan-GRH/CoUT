import re
import json
import time
from typing import Dict, List, Tuple, Any, Optional

from utils import load_config, compose_request


def extract_number_from_response(response: str) -> int:
    """
    从模型的响应中提取出估计的token数量
    
    Args:
        response: 模型的预估token数量的响应文本
        
    Returns:
        int: 提取出的token数量，如果未找到则返回300（默认值）
    """
    # 尝试匹配[[数字]]格式
    pattern1 = r"\[\[(\d+)\]\]"
    match1 = re.search(pattern1, response)
    if match1:
        return int(match1.group(1))
    
    # 尝试匹配数字词后跟tokens的格式
    pattern2 = r"(\d+)\s*tokens"
    match2 = re.search(pattern2, response, re.IGNORECASE)
    if match2:
        return int(match2.group(1))
    
    # 尝试匹配通用数字提取
    pattern3 = r"(\d+)"
    match3 = re.search(pattern3, response)
    if match3:
        return int(match3.group(1))
    
    # 如果都没有匹配到，返回默认值
    return 300  # 默认token预算


def create_budget_estimation_prompt(question: str) -> str:
    """
    创建用于预估token数量的提示词
    
    Args:
        question: 原始问题文本
        
    Returns:
        str: 预估token数量的提示词
    """
    # 创建预估token数量的提示词
    prompt = """Task: Analyze the given math problem and estimate the minimum number of tokens required 
to generate a complete and accurate solution. Your estimate should cover all necessary 
reasoning steps. Respond with ONLY a number, without any explanation or additional text.

Problem: """
    prompt += question
    
    return prompt


def add_budget_to_prompt(system_prompt: str, budget: int) -> str:
    """
    将预估的token数量添加到系统提示词中
    
    Args:
        system_prompt: 原始系统提示词
        budget: 预估的token数量
        
    Returns:
        str: 添加了token预算的系统提示词
    """
    # 将{budget}占位符替换为实际的预估token数量
    return system_prompt.replace("{budget}", str(budget))


def tale_ep_evaluate_example(task, model: str, config: Any, shot: int, example: Any, llm_client) -> Tuple[bool, int, int, int]:
    """
    使用TALE-EP方法评估单个例子
    
    Args:
        task: 任务对象
        model: 使用的模型
        config: 配置对象
        shot: few-shot示例数量
        example: 要评估的例子
        llm_client: LLM客户端
        
    Returns:
        Tuple[bool, int, int, int]: 
            - 是否正确
            - 预估阶段使用的token数
            - 解题阶段使用的token数
            - 总共使用的token数
    """
    # 步骤1：创建预估token数量的提示词
    budget_estimation_prompt = create_budget_estimation_prompt(example.question)
    
    # 步骤2：向模型发送预估请求
    start_time = time.time()
    budget_response, budget_token_count = llm_client.request(budget_estimation_prompt, model)
    end_time = time.time()
    
    # 记录预估阶段的延迟
    estimation_latency = end_time - start_time
    
    # 步骤3：从响应中提取预估的token数量
    estimated_budget = extract_number_from_response(budget_response)
    
    # 步骤4：将预估的token数量添加到系统提示词中
    updated_system_prompt = add_budget_to_prompt(config.system_prompt, estimated_budget)
    
    # 创建一个临时配置对象，使用更新后的系统提示词
    temp_config = config.model_copy()
    temp_config.system_prompt = updated_system_prompt
    
    # 步骤5：准备payload
    payload = compose_request(temp_config, shot, example.question)
    
    # 步骤6：运行推理
    start_time = time.time()
    response, solution_token_count = llm_client.request(payload, model)
    end_time = time.time()
    
    # 记录解题阶段的延迟
    solution_latency = end_time - start_time
    
    # 步骤7：检查结果
    predicted_answer = task.extract_answer(response)
    # 直接使用example.answer作为预期答案，而不是尝试进一步提取
    expected_answer = example.answer
    correct = task.equal(predicted_answer, expected_answer)
    
    # 记录详细结果，包含token数据
    task.detailed_results.append((
        example.question, 
        expected_answer, 
        predicted_answer, 
        f"[预估Token: {estimated_budget}] {response}", 
        correct,
        budget_token_count,  # 添加预估阶段token数
        solution_token_count  # 添加解题阶段token数
    ))
    
    # 计算总token数（预估阶段 + 解题阶段）
    total_token_count = budget_token_count + solution_token_count
    
    # 计算总延迟
    total_latency = estimation_latency + solution_latency
    task.latency_tracker.append(total_latency)
    
    # 记录结果
    if not correct:
        print(f"Example: {example.question}")
        print(f"预估Token数: {estimated_budget}")
        print(f"Expected: {expected_answer}, Predicted: {predicted_answer}")
        print(f"Full response: {response}")
        
    return correct, budget_token_count, solution_token_count, total_token_count
