#!/usr/bin/env python3
# check_env.py - 检查环境变量是否正确设置

import os
from dotenv import load_dotenv
import sys

def check_env():
    """检查环境变量是否正确设置"""
    print("检查环境变量...")
    
    # 检查.env文件是否存在
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(env_path):
        print(f"错误: .env文件不存在: {env_path}")
        return False
    
    print(f".env文件存在: {env_path}")
    
    # 尝试读取.env文件内容
    try:
        with open(env_path, 'r') as f:
            env_content = f.read()
        print(f".env文件内容长度: {len(env_content)} 字符")
        
        # 检查文件格式
        lines = env_content.strip().split('\n')
        print(f".env文件包含 {len(lines)} 行")
        
        for i, line in enumerate(lines):
            if '=' not in line:
                print(f"警告: 第 {i+1} 行格式不正确: {line}")
            else:
                key, value = line.split('=', 1)
                if not key.strip() or not value.strip():
                    print(f"警告: 第 {i+1} 行键或值为空: {line}")
                else:
                    print(f"第 {i+1} 行: {key.strip()} = {'*' * min(5, len(value.strip()))}...")
    except Exception as e:
        print(f"读取.env文件时出错: {str(e)}")
    
    # 加载环境变量
    load_dotenv()
    
    # 检查关键环境变量
    gemini_key = os.getenv("GEMINI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if gemini_key:
        print(f"GEMINI_API_KEY: 已设置，长度: {len(gemini_key)}")
    else:
        print("GEMINI_API_KEY: 未设置")
    
    if google_key:
        print(f"GOOGLE_API_KEY: 已设置，长度: {len(google_key)}")
    else:
        print("GOOGLE_API_KEY: 未设置")
    
    # 检查是否至少有一个API密钥
    if not gemini_key and not google_key:
        print("错误: 未找到任何API密钥")
        return False
    
    print("环境变量检查完成")
    return True

if __name__ == "__main__":
    if check_env():
        print("环境变量设置正确")
        sys.exit(0)
    else:
        print("环境变量设置有问题")
        sys.exit(1) 