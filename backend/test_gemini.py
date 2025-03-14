#!/usr/bin/env python3
# test_gemini.py - 测试Gemini API是否能正常工作

import os
from dotenv import load_dotenv
import google.generativeai as genai

# 载入环境变量
load_dotenv()

def test_gemini_api():
    """测试Gemini API是否能正常工作"""
    print("开始测试Gemini API...")
    
    # 获取API密钥
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("错误: 未找到API密钥，请检查.env文件")
        return False
    
    # 打印API密钥信息（不显示完整密钥）
    print(f"API密钥前5个字符: {api_key[:5]}, 长度: {len(api_key)}")
    
    try:
        # 配置API
        genai.configure(api_key=api_key)
        print("Gemini API配置成功")
        
        # 创建模型实例
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("成功创建模型实例")
        
        # 发送简单请求
        prompt = "请用JSON格式返回一个包含3个简单数学题的列表，格式为: {\"questions\": [{\"id\": 1, \"content\": \"题目内容\", \"answer\": \"答案\"}]}"
        print(f"发送测试请求: {prompt}")
        
        response = model.generate_content(prompt)
        print("成功收到响应")
        print(f"响应内容: {response.text[:200]}...")
        
        print("Gemini API测试成功!")
        return True
    
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_gemini_api() 