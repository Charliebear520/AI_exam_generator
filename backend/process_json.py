#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用JSON處理工具函數

包含處理JSON序列化問題的函數和類，可在項目中的任何地方使用
"""

import json
import re
import sys
from typing import Any, Dict, List, Union, Optional

class EnhancedJSONEncoder(json.JSONEncoder):
    """增強的JSON編碼器，支持各種Python類型的序列化
    
    處理的類型包括:
    - None值
    - NumPy數組和標量
    - 日期和時間對象
    - 集合和元組
    - 具有特殊方法的對象
    - 其他不可直接序列化的類型
    """
    def default(self, obj: Any) -> Any:
        # 處理None值
        if obj is None:
            return ""
        
        # 處理NumPy相關類型
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):
                return obj.tolist()
            # 處理可能的NumPy陣列條件檢查
            if hasattr(obj, 'any') and hasattr(obj, 'all') and hasattr(obj, 'shape'):
                try:
                    return np.asarray(obj).tolist()
                except:
                    pass
        except (ImportError, NameError):
            pass  # NumPy不可用，跳過
        
        # 處理集合和元組
        if isinstance(obj, (set, tuple)):
            return list(obj)
        
        # 處理日期和時間
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        
        # 處理可轉換為字典的對象
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        
        # 處理有__dict__屬性的對象
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        
        # 處理可迭代對象(但不是字符串或字節)
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
            return list(obj)
        
        # 嘗試默認處理
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)  # 轉換為字符串作為最後手段

def clean_for_json(obj: Any) -> Any:
    """深度清理對象，使其可以被JSON序列化
    
    處理各種類型，包括嵌套的字典和列表
    
    Args:
        obj: 需要清理的對象
        
    Returns:
        可以被JSON序列化的對象
    """
    # 處理None值
    if obj is None:
        return ""
    
    # 處理NumPy相關類型
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        # 處理NumPy類數組
        if hasattr(obj, 'shape') or hasattr(obj, 'dtype'):
            try:
                return np.asarray(obj).tolist()
            except:
                pass
    except (ImportError, NameError):
        pass  # NumPy不可用，跳過
    
    # 處理字典 - 遞歸清理鍵和值
    if isinstance(obj, dict):
        return {
            str(k) if k is not None else "None": clean_for_json(v)
            for k, v in obj.items()
        }
    
    # 處理列表和元組 - 遞歸清理每個元素
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    
    # 處理集合
    if isinstance(obj, set):
        return [clean_for_json(item) for item in obj]
    
    # 處理日期和時間
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    
    # 檢查是否為標準JSON類型
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # 處理其他對象類型
    try:
        # 嘗試轉換為字典
        if hasattr(obj, 'to_dict'):
            return clean_for_json(obj.to_dict())
        # 嘗試使用__dict__
        elif hasattr(obj, '__dict__'):
            return clean_for_json(obj.__dict__)
        # 嘗試轉換可迭代對象
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            return [clean_for_json(item) for item in obj]
        else:
            # 轉換為字符串作為最後手段
            return str(obj)
    except Exception as e:
        # 如果所有嘗試都失敗，返回一個安全的字符串
        return f"[無法序列化: {type(obj).__name__}]"

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """安全地將對象轉換為JSON字符串
    
    使用增強的編碼器和清理函數確保對象可以被序列化
    
    Args:
        obj: 要序列化的對象
        **kwargs: 傳遞給json.dumps的其他參數
        
    Returns:
        JSON字符串
    """
    try:
        # 首先嘗試直接序列化
        return json.dumps(obj, cls=EnhancedJSONEncoder, **kwargs)
    except (TypeError, OverflowError) as e:
        # 如果失敗，嘗試清理後再序列化
        cleaned_obj = clean_for_json(obj)
        try:
            return json.dumps(cleaned_obj, **kwargs)
        except Exception as e2:
            # 如果還是失敗，返回一個錯誤對象
            error_obj = {
                "error": "JSON序列化失敗",
                "message": str(e2),
                "original_error": str(e)
            }
            return json.dumps(error_obj)

def safe_json_loads(json_str: str, default: Optional[Any] = None) -> Any:
    """安全地解析JSON字符串
    
    處理各種常見的JSON格式錯誤
    
    Args:
        json_str: JSON字符串
        default: 解析失敗時返回的默認值
        
    Returns:
        解析後的對象或默認值
    """
    if not json_str or not isinstance(json_str, str):
        return default if default is not None else {}
    
    # 移除Markdown代碼塊符號
    json_str = re.sub(r'```(?:json)?\s*|\s*```', '', json_str)
    
    try:
        # 嘗試直接解析
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # 如果失敗，嘗試修復常見問題
        
        # 1. 替換單引號為雙引號
        fixed_str = json_str.replace("'", '"')
        
        # 2. 處理沒有引號的屬性名
        fixed_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', fixed_str)
        fixed_str = re.sub(r'(^|\n)(\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1\2"\3"\4', fixed_str)
        
        # 3. 修復逗號問題
        fixed_str = re.sub(r',\s*}', '}', fixed_str)
        fixed_str = re.sub(r',\s*]', ']', fixed_str)
        
        # 4. 修復括號不匹配
        open_braces = fixed_str.count('{')
        close_braces = fixed_str.count('}')
        if open_braces > close_braces:
            fixed_str += "}" * (open_braces - close_braces)
        
        try:
            # 嘗試解析修復後的字符串
            return json.loads(fixed_str)
        except json.JSONDecodeError:
            # 如果還是失敗，返回默認值
            return default if default is not None else {}

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """從文本中提取JSON對象
    
    處理包含在Markdown代碼塊中或混合在其他文本中的JSON
    
    Args:
        text: 可能包含JSON的文本
        
    Returns:
        提取到的JSON對象或空字典
    """
    if not text or not isinstance(text, str):
        return {}
    
    # 移除Markdown代碼塊符號
    text = re.sub(r'```(?:json)?\s*|\s*```', '', text)
    
    # 尋找JSON對象
    json_pattern = r'(\{[\s\S]*\})'
    match = re.search(json_pattern, text)
    
    if match:
        json_str = match.group(1)
        # 使用安全解析函數
        return safe_json_loads(json_str, {})
    
    return {}

if __name__ == "__main__":
    # 測試代碼
    test_obj = {
        "string": "測試字符串",
        "number": 42,
        "list": [1, 2, 3],
        "nested": {
            "a": None,
            "b": [4, 5, 6]
        }
    }
    
    # 測試序列化和反序列化
    json_str = safe_json_dumps(test_obj, ensure_ascii=False, indent=2)
    print("序列化結果:")
    print(json_str)
    
    parsed = safe_json_loads(json_str)
    print("\n反序列化結果:")
    print(parsed) 