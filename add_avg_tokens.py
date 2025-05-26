#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import sys
from glob import glob
import re

def find_latest_json_file(directory):
    """查找指定目录下最新的JSON文件"""
    # 使用正则表达式匹配gsm8k模型的详细JSON文件
    pattern = re.compile(r'\d{8}_\d{6}_gsm8k.*_tale_ep_detailed\.json$')
    json_files = []
    
    # 递归查找所有符合条件的文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                json_files.append(os.path.join(root, file))
    
    # 按照修改时间排序，取最新的文件
    if json_files:
        return max(json_files, key=os.path.getmtime)
    return None

def add_avg_second_query_tokens(json_file):
    """向JSON文件添加avg_second_query_tokens字段"""
    try:
        print(f"处理文件: {json_file}")
        
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否是tale_ep方法
        if data.get('prompt') != 'tale_ep':
            print(f"警告: 该文件不是tale_ep方法，prompt={data.get('prompt')}")
            return False
            
        # 计算平均second_query_tokens
        if 'detailed_results' in data and data['detailed_results']:
            second_query_tokens = []
            for result in data['detailed_results']:
                if 'second_query_tokens' in result:
                    second_query_tokens.append(result['second_query_tokens'])
            
            # 计算平均值
            if second_query_tokens:
                avg_second_query_tokens = sum(second_query_tokens) / len(second_query_tokens)
                
                # 在顶层添加字段
                data['avg_second_query_tokens'] = avg_second_query_tokens
                data['Avg_second_query_tokens'] = avg_second_query_tokens
                
                # 保存修改后的JSON
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"已成功添加avg_second_query_tokens字段到文件: {json_file}")
                print(f"值为: {avg_second_query_tokens}")
                return True
            else:
                print(f"警告: 未在文件中找到second_query_tokens字段")
                return False
        
        print("未找到detailed_results字段")
        return False
    
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

if __name__ == "__main__":
    # 判断是否提供了文件路径参数
    if len(sys.argv) > 1:
        # 使用命令行参数中指定的文件路径
        json_file = sys.argv[1]
        if os.path.exists(json_file):
            add_avg_second_query_tokens(json_file)
        else:
            print(f"错误: 指定的文件不存在: {json_file}")
    else:
        # 没有提供文件路径，则查找最新的JSON文件
        results_dir = "./results"
        latest_json = find_latest_json_file(results_dir)
        
        if latest_json:
            print(f"找到最新的JSON文件: {latest_json}")
            add_avg_second_query_tokens(latest_json)
        else:
            print("未找到符合条件的JSON文件或请指定文件路径: python add_avg_tokens.py <json文件路径>") 