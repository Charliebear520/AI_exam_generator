#!/usr/bin/env python3
# debug_gemini.py - 调试Gemini API响应

import os
import json
import time
from dotenv import load_dotenv
import google.generativeai as genai

# 载入环境变量
load_dotenv()

def debug_gemini_api(text_sample, save_response=True):
    """
    测试Gemini API并保存原始响应
    
    Args:
        text_sample: 要处理的文本样本
        save_response: 是否保存响应到文件
    """
    print("开始调试Gemini API...")
    
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
        
        # 创建模型实例 - 使用正确的模型
        models = ['gemini-2.0-flash']  # 只使用可用的模型
        
        for model_name in models:
            try:
                print(f"\n测试模型: {model_name}")
                model = genai.GenerativeModel(model_name)
                
                # 使用更简洁的提示词
                prompt = f"""
                分析以下考试题目文本，提取题目信息并以JSON格式返回。

                JSON格式示例：
                {{
                  "questions": [
                    {{
                      "id": 1,
                      "content": "题干",
                      "options": {{"A": "选项A", "B": "选项B"}},
                      "answer": "A",
                      "explanation": "解析"
                    }}
                  ]
                }}

                考试题目文本：
                {text_sample}

                只返回JSON格式数据，不要有其他文字。
                """
                
                print(f"发送请求到 {model_name}...")
                response = model.generate_content(prompt)
                print(f"成功收到 {model_name} 响应")
                
                # 保存原始响应
                if save_response:
                    timestamp = int(time.time())
                    filename = f"gemini_{model_name}_response_{timestamp}.txt"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    print(f"原始响应已保存到: {filename}")
                
                # 尝试解析JSON
                try:
                    # 尝试从代码块中提取JSON
                    import re
                    json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(1)
                    else:
                        json_match = re.search(r'({.*})', response.text, re.DOTALL)
                        if json_match:
                            json_text = json_match.group(1)
                        else:
                            json_text = response.text
                    
                    # 解析JSON
                    result = json.loads(json_text)
                    print(f"{model_name} 返回的JSON有效")
                    
                    # 检查是否包含questions字段
                    if "questions" in result:
                        print(f"成功解析出 {len(result['questions'])} 道题目")
                    else:
                        print("JSON中没有questions字段")
                    
                except Exception as e:
                    print(f"{model_name} 返回的JSON无效: {str(e)}")
            
            except Exception as model_error:
                print(f"测试 {model_name} 时出错: {str(model_error)}")
        
        print("\n调试完成")
        return True
    
    except Exception as e:
        print(f"调试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 示例文本 - 可以替换为实际的PDF提取文本
    sample_text = """
    1. 以下哪项不是Python的基本数据类型？
    A. 整数 (int)
    B. 浮点数 (float)
    C. 字符串 (string)
    D. 数组 (array)
    
    2. 在Python中，以下哪个语句正确创建了一个空列表？
    A. list = []
    B. list = {}
    C. list = ()
    D. list = None
    
    3. Python中的字典是什么类型的数据结构？
    A. 有序集合
    B. 键值对集合
    C. 数组
    D. 链表
    """
    
    debug_gemini_api(sample_text) 