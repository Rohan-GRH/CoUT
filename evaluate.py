import argparse
import csv
import os
import json
import time
from datetime import datetime
import yaml  # 导入处理YAML的库

from llm_client import LLMClient
from tasks.gsm8k import GSM8K
from tasks.aqua import AQUA
from tasks.svamp import SVAMP
from tasks.mathqa import MathQA


# 导入TALE-EP任务类
from tasks.gsm8k_tale import GSM8KTale
from tasks.aqua_tale import AQUATale
from tasks.mathqa_tale import MathQATale
from tasks.svamp_tale import SVAMPTale
from utils import average, nth_percentile, load_config


MODEL_MAPPING = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "sonnet": "claude-3-5-sonnet-20240620",
    "o3-mini": "o3-mini",
    "qwen-qwq-32b": "Qwen/QwQ-32B",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", 
        choices=[
            "gsm8k", "aqua", "svamp", "mathqa"
        ],
        help="任务名称"
    )
    parser.add_argument("--model", default="claude3.5")
    parser.add_argument(
        "--prompt",
        choices=["baseline", "cod", "cot", "CoUT", "tale_ep"],
        default="CoUT",
        help="Prompting strategy",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=None,
        help="Number of fewshot to be included, by default, include all fewshot examples",
    )
    parser.add_argument(
        "--max_samples", 
        default=None, 
        help="Maximum number of samples to evaluate. Use 'max' to evaluate all samples."
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Base url for llm model endpoint",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for model access, will use api keys in environment variables for openai and claude models.",
    )




    args = parser.parse_args()
    llm_client = LLMClient(args.url, args.api_key)
    
    # 处理 max_samples 参数
    max_samples = args.max_samples
    if max_samples is not None:
        if max_samples == 'max' or max_samples.lower() == 'max':
            # 设置为None表示使用全部数据
            max_samples = None
        else:
            try:
                max_samples = int(max_samples)
            except ValueError:
                print(f"警告: max_samples值'{max_samples}'无效，应为整数或'max'。将使用全部数据。")
                max_samples = None
    

    
    # 如果使用tale_ep评估方法，对于支持的任务使用对应的TALE任务类
    tale_ep_mode = False  # 添加标记，用于判断是否使用TALE-EP方法
    if args.prompt == "tale_ep":
        tale_ep_mode = True  # 设置标记
        match args.task:
            case "gsm8k":
                task = GSM8KTale(llm_client)
            case "aqua":
                task = AQUATale(llm_client)
            case "svamp":
                task = SVAMPTale(llm_client)
            case "mathqa":
                task = MathQATale(llm_client)
            case _:
                print(f"警告: 任务 {args.task} 不支持TALE-EP评估方法，将使用标准任务类。")
                args.prompt = "cod"  # 回退到COD方法
                tale_ep_mode = False  # 重置标记
                # 使用标准任务类
                match args.task:
                    case "gsm8k":
                        task = GSM8K(llm_client)
                    case "aqua":
                        task = AQUA(llm_client)
                    case "svamp":
                        task = SVAMP(llm_client)
                    case "mathqa":
                        task = MathQA(llm_client)
                    case _:
                        raise ValueError("Invalid task")
    else:
        # 使用标准任务类
        match args.task:
            case "gsm8k":
                task = GSM8K(llm_client)
            case "aqua":
                task = AQUA(llm_client)
            case "svamp":
                task = SVAMP(llm_client)
            case "mathqa":
                task = MathQA(llm_client)
            case _:
                raise ValueError("Invalid task")

    model = MODEL_MAPPING.get(args.model, args.model)
    
    # 如果是TALE-EP方法，获取更详细的评估结果
    if tale_ep_mode:
        evaluation_result = task.evaluate(model, args.prompt, args.shot, max_samples)
        if isinstance(evaluation_result, dict):
            accuracy = evaluation_result["accuracy"]
            # 保存第一次和第二次回复的平均token数
            estimation_token_avg = evaluation_result.get("estimation_token_avg", 0)
            solution_token_avg = evaluation_result.get("solution_token_avg", 0)
            
            # 直接从详细结果中计算第二次查询的平均token数
            second_query_tokens_list = []
            if hasattr(task, "detailed_results") and task.detailed_results:
                for result in task.detailed_results:
                    if len(result) > 6:  # 确保结果包含token信息
                        second_query_tokens_list.append(result[6])  # 第二次查询token
            
            # 如果有详细结果，直接计算平均值
            avg_second_query_tokens = sum(second_query_tokens_list) / len(second_query_tokens_list) if second_query_tokens_list else solution_token_avg
            
            results = [
                [
                    "Accuracy",
                    "First Query Avg Token #",  # 第一次查询的平均token数
                    "Second Query Avg Token #", # 第二次查询的平均token数
                    "Avg second_query_tokens",  # 添加与用户需求一致的命名
                    "Total Avg Token #",        # 总平均token数
                    "Average Latency (s)",
                    "P90 Latency (s)",
                    "P95 Latency (s)",
                    "P99 Latency (s)",
                ],
                [
                    accuracy,
                    estimation_token_avg,       # 添加第一次查询的平均token数
                    solution_token_avg,         # 添加第二次查询的平均token数
                    avg_second_query_tokens,    # 使用直接计算的平均值
                    average(task.token_count_tracker),
                    average(task.latency_tracker),
                    nth_percentile(task.latency_tracker, 0.9),
                    nth_percentile(task.latency_tracker, 0.95),
                    nth_percentile(task.latency_tracker, 0.99),
                ],
            ]
        else:
            # 如果不是字典形式的结果，则用标准处理方式
            accuracy = evaluation_result
            results = [
                [
                    "Accuracy",
                    "Avg Token #",
                    "Average Latency (s)",
                    "P90 Latency (s)",
                    "P95 Latency (s)",
                    "P99 Latency (s)",
                ],
                [
                    accuracy,
                    average(task.token_count_tracker),
                    average(task.latency_tracker),
                    nth_percentile(task.latency_tracker, 0.9),
                    nth_percentile(task.latency_tracker, 0.95),
                    nth_percentile(task.latency_tracker, 0.99),
                ],
            ]
    else:
        # 非TALE-EP方法的标准处理
        accuracy = task.evaluate(model, args.prompt, args.shot, max_samples)
        results = [
            [
                "Accuracy",
                "Avg Token #",
                "Average Latency (s)",
                "P90 Latency (s)",
                "P95 Latency (s)",
                "P99 Latency (s)",
            ],
            [
                accuracy,
                average(task.token_count_tracker),
                average(task.latency_tracker),
                nth_percentile(task.latency_tracker, 0.9),
                nth_percentile(task.latency_tracker, 0.95),
                nth_percentile(task.latency_tracker, 0.99),
            ],
        ]
    
    # 打印总体结果
    print("\n=== Overall Results ===")
    for i in range(len(results[0])):
        print(f"{results[0][i]}: {results[1][i]}")
        
    # 如果是TALE-EP方法，额外打印两次查询的token信息
    if tale_ep_mode and isinstance(evaluation_result, dict):
        print("\n=== TALE-EP Token Information ===")
        print(f"First Query (Estimation) Avg Token: {estimation_token_avg:.2f}")
        print(f"Second Query (Solution) Avg Token: {solution_token_avg:.2f}")
        print(f"Avg second_query_tokens: {avg_second_query_tokens:.2f}")  # 使用直接计算的平均值
        print(f"Total Avg Token: {average(task.token_count_tracker):.2f}")
    
    # 打印详细结果
    print("\n=== Detailed Results ===")
    if hasattr(task, "detailed_results") and task.detailed_results:
        # 检查是否是TALE-EP方法（详细结果中是否包含token数据）
        if tale_ep_mode and len(task.detailed_results) > 0 and len(task.detailed_results[0]) > 5:
            for i, (question, expected, predicted, response, is_correct, first_query_tokens, second_query_tokens) in enumerate(task.detailed_results):
                print(f"\nSample {i+1}:")
                print(f"Question: {question}")
                print(f"Expected: {expected}")
                print(f"Predicted: {predicted}")
                print(f"Full response: {response}")
                print(f"Status: {'✓ Correct' if is_correct else '✗ Incorrect'}")
                print(f"First Query Tokens: {first_query_tokens}")
                print(f"Second Query Tokens: {second_query_tokens}")
                print(f"Total Tokens: {first_query_tokens + second_query_tokens}")
        else:
            for i, (question, expected, predicted, response, is_correct) in enumerate(task.detailed_results):
                print(f"\nSample {i+1}:")
                print(f"Question: {question}")
                print(f"Expected: {expected}")
                print(f"Predicted: {predicted}")
                print(f"Full response: {response}")
                print(f"Status: {'✓ Correct' if is_correct else '✗ Incorrect'}")
    else:
        print("No detailed results available.")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    # 使用当前日期时间作为文件名的一部分
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_date = datetime.now().strftime("%Y%m%d")  # 提取当前日期，用于创建文件夹

    # 创建日期文件夹
    date_dir = os.path.join("./results", current_date)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
        
    # 创建任务(数据集)子文件夹
    task_dir = os.path.join(date_dir, args.task)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    
    # 为MMLU添加学科信息到文件名
    task_identifier = args.task
    if args.task == "mmlu" and args.mmlu_subject:
        task_identifier = f"{args.task}_{args.mmlu_subject}"

    model_name = args.model.split(":")[1] if ":" in args.model else args.model
    model_name = model_name.replace("/", "_")
    fname = (
        f"{current_time}_{task_identifier}-{model_name}-{args.prompt}-{args.shot}"
        if args.shot
        else f"{current_time}_{task_identifier}-{model_name}-{args.prompt}"
    )
    
    # 保存总体结果到CSV (放入任务文件夹)
    with open(os.path.join(task_dir, f"{fname}.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)
    
    # 读取YAML配置文件获取prompt信息
    config = load_config(args.task, args.prompt)
    prompt_description = config.system_prompt

    # 保存详细结果到JSON
    detailed_results = {
        "timestamp": current_time,
        "task": args.task,
        "model": model_name,
        "prompt": args.prompt,
        "shot": args.shot,
        "max_samples": args.max_samples,
        "overall_results": dict(zip(results[0], results[1])),
        "prompt_description": prompt_description,
    }
    
    # 如果是TALE-EP方法，确保添加第二次查询的平均token数
    if tale_ep_mode and isinstance(evaluation_result, dict):
        # 直接添加到JSON的最顶层，确保它会出现在文件中
        detailed_results.update({
            "Avg_second_query_tokens": float(avg_second_query_tokens),
            "avg_second_query_tokens": float(avg_second_query_tokens),
            "平均第二次查询tokens": float(avg_second_query_tokens),
            "second_query_avg_token": float(avg_second_query_tokens),
            "average_second_query_tokens": float(avg_second_query_tokens)
        })
        
        # 确保也添加到overall_results中
        if "overall_results" in detailed_results:
            detailed_results["overall_results"]["Avg_second_query_tokens"] = float(avg_second_query_tokens)
    
    # 如果是TALE-EP方法，添加更详细的token信息
    if tale_ep_mode and isinstance(evaluation_result, dict):
        detailed_results["tale_ep_token_info"] = {
            "first_query_avg_token": float(estimation_token_avg),
            "second_query_avg_token": float(solution_token_avg),
            "avg_second_query_tokens": float(avg_second_query_tokens),  # 使用直接计算的平均值
            "total_avg_token": float(average(task.token_count_tracker))
        }
        
        # 添加"TALE-EP 统计"部分的信息到JSON文件中
        detailed_results["tale_ep_stats"] = {
            "预估阶段平均Token": float(estimation_token_avg),
            "解题阶段平均Token": float(solution_token_avg),
            "Avg second_query_tokens": float(avg_second_query_tokens),  # 使用直接计算的平均值
            "总平均Token": float(average(task.token_count_tracker))
        }
    
  
    
    # 如果有详细结果，则添加到JSON中
    if hasattr(task, "detailed_results") and task.detailed_results:
        # 检查是否是TALE-EP方法（详细结果中是否包含token数据）
        if tale_ep_mode and len(task.detailed_results) > 0 and len(task.detailed_results[0]) > 5:
            # TALE-EP方法的详细结果，包含每个样本的token数据
            detailed_results["detailed_results"] = [
                {
                    "question": question,
                    "expected": expected,
                    "predicted": predicted,
                    "response": response,
                    "is_correct": is_correct,
                    "first_query_tokens": first_query_tokens,  # 预估阶段token数
                    "second_query_tokens": second_query_tokens  # 解题阶段token数
                }
                for question, expected, predicted, response, is_correct, first_query_tokens, second_query_tokens in task.detailed_results
            ]
            
            # 保存每个样本的token数据统计
            if len(task.detailed_results) > 0:
                sample_token_stats = []
                for _, _, _, _, _, first_tokens, second_tokens in task.detailed_results:
                    sample_token_stats.append({
                        "first_query_tokens": first_tokens,
                        "second_query_tokens": second_tokens,
                        "total_tokens": first_tokens + second_tokens
                    })
                detailed_results["sample_token_stats"] = sample_token_stats
        else:
            # 标准方法的详细结果
            detailed_results["detailed_results"] = [
                {
                    "question": question,
                    "expected": expected,
                    "predicted": predicted,
                    "response": response,
                    "is_correct": is_correct
                }
                for question, expected, predicted, response, is_correct in task.detailed_results
            ]
    
    # 保存JSON文件到任务文件夹
    with open(os.path.join(task_dir, f"{fname}_detailed.json"), "w", encoding="utf-8") as f:
        # 在序列化之前，确保avg_second_query_tokens字段一定存在于JSON最顶层
        if tale_ep_mode and isinstance(evaluation_result, dict):
            detailed_results["Avg_second_query_tokens"] = float(avg_second_query_tokens)  # 添加一个首字母大写的版本
            detailed_results["avg_second_query_tokens"] = float(avg_second_query_tokens)  # 原始小写版本
        
        # 先转换为JSON字符串
        json_str = json.dumps(detailed_results, ensure_ascii=False, indent=2)
        
        # 如果是TALE-EP模式，直接在JSON字符串中添加avg_second_query_tokens字段
        if tale_ep_mode and isinstance(evaluation_result, dict):
            # 定位到第一个大括号之后的位置
            insert_pos = json_str.find('{') + 1
            # 构造要插入的字段
            insert_str = f'\n  "avg_second_query_tokens": {float(avg_second_query_tokens)},'
            # 插入字段
            json_str = json_str[:insert_pos] + insert_str + json_str[insert_pos:]
        
        # 写入文件
        f.write(json_str)

    # 打印文件保存位置
    print(f"\n=== Results saved to ===")
    print(f"CSV: {os.path.join(task_dir, f'{fname}.csv')}")
    print(f"JSON: {os.path.join(task_dir, f'{fname}_detailed.json')}")
