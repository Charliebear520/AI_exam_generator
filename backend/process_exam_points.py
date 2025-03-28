"""
考點預處理工具

該模塊提供功能用於處理和優化法律考點的格式和內容，包括：
1. 考點格式化: 清理和規範化考點文本
2. 長考點拆分: 將複合考點拆分為獨立的法律概念
3. 考點層級建立: 組織考點為層級結構
4. 考點標準化: 根據法律術語和標準確保考點一致性
"""

import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Set
import traceback

# 載入環境變量
load_dotenv()

# 設置API金鑰
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def format_exam_point(exam_point: str) -> str:
    """
    基本的考點格式化處理
    
    Args:
        exam_point: 原始考點字符串
    
    Returns:
        格式化後的考點
    """
    if not exam_point or not isinstance(exam_point, str):
        return ""
    
    # 清理空白和特殊字符
    formatted = exam_point.strip()
    
    # 移除多餘的空格
    formatted = re.sub(r'\s+', ' ', formatted)
    
    # 移除不必要的標點符號
    formatted = re.sub(r'^[「『【《\s]+|[」』】》\s]+$', '', formatted)
    
    # 簡化法條表示
    formatted = re.sub(r'第(\d+)條第(\d+)項第(\d+)款', r'§\1(\2)(\3)', formatted)
    formatted = re.sub(r'第(\d+)條第(\d+)項', r'§\1(\2)', formatted)
    formatted = re.sub(r'第(\d+)條', r'§\1', formatted)
    
    return formatted

def is_complex_exam_point(exam_point: str) -> bool:
    """
    判斷一個考點是否是複合考點需要拆分
    
    Args:
        exam_point: 考點字符串
    
    Returns:
        是否是複合考點
    """
    if not exam_point:
        return False
    
    # 檢查字符長度，超過一定長度可能是複合考點
    if len(exam_point) > 30:
        return True
    
    # 檢查特定分隔符號
    separators = ['、', '，', '；', '：', '：', '/', '及', '與', '和']
    for sep in separators:
        if sep in exam_point:
            # 不計算法條編號中的分隔符
            if not re.search(r'第\d+條.*' + sep, exam_point):
                return True
    
    return False

def split_exam_point(exam_point: str) -> List[str]:
    """
    使用Gemini將複合考點拆分為多個簡單考點
    
    Args:
        exam_point: 複合考點字符串
    
    Returns:
        拆分後的考點列表
    """
    if not is_complex_exam_point(exam_point):
        return [exam_point]
    
    try:
        # 使用Gemini進行考點拆分
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        你是一位專業的法律考試專家，請將以下法律考點拆分為獨立的、簡潔的法律概念。
        每個概念應該表示一個具體、明確的法律知識點，而不是籠統的分類。
        
        原考點：
        {exam_point}
        
        請將這個考點拆分為多個獨立的法律概念，每個概念應該：
        1. 足夠具體（例如"無權代理"而不是"代理"）
        2. 使用標準法律術語
        3. 一般不超過10個字
        4. 不應該包含過多的條件、限定或例外情況
        5. 每個概念應該是一個完整的法律知識點
        
        請直接以列表形式輸出拆分後的法律概念，每行一個，不要有序號或其他標記。
        """
        
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # 解析結果
        split_points = [line.strip() for line in result_text.split('\n') if line.strip()]
        
        # 處理格式
        formatted_points = [format_exam_point(point) for point in split_points]
        
        # 去除空值和重複
        final_points = list(set([p for p in formatted_points if p]))
        
        print(f"將考點「{exam_point}」拆分為: {final_points}")
        return final_points
    except Exception as e:
        print(f"拆分考點時出錯：{str(e)}")
        traceback.print_exc()
        # 如果失敗則返回原始考點
        return [exam_point]

def process_exam_point_batch(exam_points: List[str]) -> Dict[str, List[str]]:
    """
    批量處理考點，拆分複合考點並返回映射關係
    
    Args:
        exam_points: 考點列表
    
    Returns:
        原考點到拆分考點的映射字典
    """
    mapping = {}
    for point in exam_points:
        formatted = format_exam_point(point)
        if is_complex_exam_point(formatted):
            split_points = split_exam_point(formatted)
            mapping[point] = split_points
        else:
            mapping[point] = [formatted]
    
    return mapping

def normalize_exam_points(points: List[str]) -> List[str]:
    """
    標準化考點列表，處理同義詞和標準術語
    
    Args:
        points: 考點列表
    
    Returns:
        標準化後的考點列表
    """
    # 常見的術語標準化映射
    standardization_map = {
        # 民法相關
        "債權讓與": "債權讓與",
        "債權轉讓": "債權讓與",
        "所有權移轉": "所有權移轉",
        "所有權讓與": "所有權移轉",
        "無因管理": "無因管理",
        # 刑法相關
        "緊急避難": "緊急避難",
        "避難行為": "緊急避難",
        "自救行為": "緊急避難",
        "正當防衛": "正當防衛",
        "防衛行為": "正當防衛",
        # 法條簡化
        "民法第184條": "民法§184",
        "刑法第14條": "刑法§14"
    }
    
    # 應用標準化
    normalized = []
    for point in points:
        # 查找完全匹配
        if point in standardization_map:
            normalized.append(standardization_map[point])
        else:
            # 查找部分匹配
            found = False
            for k, v in standardization_map.items():
                if k in point or point in k:
                    normalized.append(v)
                    found = True
                    break
            
            if not found:
                normalized.append(point)
    
    # 去重
    return list(set(normalized))

def organize_exam_points_hierarchy(points: List[str]) -> Dict:
    """
    將考點組織為層級結構
    
    Args:
        points: 考點列表
    
    Returns:
        考點的層級結構
    """
    hierarchy = {
        "民法": {
            "總則": [],
            "物權": [],
            "債編": [],
            "親屬": [],
            "繼承": []
        },
        "刑法": {
            "總則": [],
            "分則": []
        },
        "行政法": [],
        "憲法": [],
        "商事法": {
            "公司法": [],
            "證券交易法": [],
            "票據法": []
        },
        "訴訟法": {
            "民事訴訟法": [],
            "刑事訴訟法": [], 
            "行政訴訟法": []
        },
        "其他": []
    }
    
    # 簡單的關鍵詞匹配分類
    for point in points:
        if "民法" in point or "契約" in point or "債權" in point or "侵權" in point:
            if "物權" in point or "所有權" in point or "抵押權" in point:
                hierarchy["民法"]["物權"].append(point)
            elif "債" in point or "契約" in point or "侵權" in point:
                hierarchy["民法"]["債編"].append(point)
            elif "親屬" in point or "婚姻" in point:
                hierarchy["民法"]["親屬"].append(point)
            elif "繼承" in point or "遺產" in point:
                hierarchy["民法"]["繼承"].append(point)
            else:
                hierarchy["民法"]["總則"].append(point)
        elif "刑法" in point or "犯罪" in point:
            if "故意" in point or "過失" in point or "正當防衛" in point or "緊急避難" in point:
                hierarchy["刑法"]["總則"].append(point)
            else:
                hierarchy["刑法"]["分則"].append(point)
        elif "行政法" in point or "行政處分" in point:
            hierarchy["行政法"].append(point)
        elif "憲法" in point or "基本權" in point:
            hierarchy["憲法"].append(point)
        elif "公司" in point or "票據" in point or "證券" in point:
            if "公司" in point:
                hierarchy["商事法"]["公司法"].append(point)
            elif "證券" in point:
                hierarchy["商事法"]["證券交易法"].append(point)
            elif "票據" in point:
                hierarchy["商事法"]["票據法"].append(point)
            else:
                hierarchy["商事法"].append(point)
        elif "訴訟" in point or "程序" in point:
            if "民事" in point:
                hierarchy["訴訟法"]["民事訴訟法"].append(point)
            elif "刑事" in point:
                hierarchy["訴訟法"]["刑事訴訟法"].append(point)
            elif "行政" in point:
                hierarchy["訴訟法"]["行政訴訟法"].append(point)
            else:
                hierarchy["訴訟法"].append(point)
        else:
            hierarchy["其他"].append(point)
    
    return hierarchy

def update_question_with_processed_exam_points(question: Dict, mapping: Dict[str, List[str]]) -> Dict:
    """
    使用處理後的考點更新題目
    
    Args:
        question: 題目數據
        mapping: 考點映射關係
    
    Returns:
        更新後的題目
    """
    original_exam_point = question.get("exam_point", "")
    if not original_exam_point:
        return question
    
    # 使用映射更新考點
    if original_exam_point in mapping:
        processed_points = mapping[original_exam_point]
        
        # 多個考點時，使用第一個作為主考點，其他添加到關鍵詞
        if processed_points:
            question["exam_point"] = processed_points[0]
            
            # 將其餘考點添加到關鍵詞
            keywords = question.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            
            for point in processed_points[1:]:
                if point not in keywords:
                    keywords.append(point)
            
            question["keywords"] = keywords
    
    return question

def process_and_update_vector_store():
    """
    處理向量庫中的全部考點並更新
    """
    from vector_store import vector_store
    
    try:
        # 獲取所有考點
        all_exam_points = vector_store.get_all_exam_points()
        if not all_exam_points:
            print("未找到任何考點")
            return
        
        print(f"獲取到 {len(all_exam_points)} 個考點")
        
        # 批量處理考點
        mapping = process_exam_point_batch(all_exam_points)
        print(f"完成 {len(mapping)} 個考點的處理")
        
        # 獲取所有題目進行更新
        all_data = vector_store.collection.get()
        if not all_data or not all_data.get("metadatas"):
            print("未找到任何題目")
            return
        
        print(f"開始更新 {len(all_data['ids'])} 個題目的考點")
        
        # 更新題目
        for i, (doc_id, metadata) in enumerate(zip(all_data["ids"], all_data["metadatas"])):
            original_exam_point = metadata.get("exam_point", "")
            if original_exam_point and original_exam_point in mapping:
                # 獲取處理後的考點
                processed_points = mapping[original_exam_point]
                
                if processed_points:
                    # 更新主考點
                    metadata["exam_point"] = processed_points[0]
                    
                    # 將其餘考點添加到關鍵詞
                    keywords = metadata.get("keywords", [])
                    if isinstance(keywords, str):
                        try:
                            keywords = json.loads(keywords)
                        except:
                            keywords = []
                    
                    if not isinstance(keywords, list):
                        keywords = []
                    
                    for point in processed_points[1:]:
                        if point and point not in keywords:
                            keywords.append(point)
                    
                    # 更新關鍵詞，確保格式正確
                    metadata["keywords"] = json.dumps(keywords) if keywords else "[]"
                    
                    # 更新向量庫中的元數據
                    vector_store.collection.update(
                        ids=[doc_id],
                        metadatas=[metadata]
                    )
        
        print("完成考點處理和更新")
        
        # 生成並保存考點層級結構
        all_processed_points = []
        for points_list in mapping.values():
            all_processed_points.extend(points_list)
        
        # 標準化考點
        normalized_points = normalize_exam_points(all_processed_points)
        
        # 組織層級結構
        hierarchy = organize_exam_points_hierarchy(normalized_points)
        
        # 保存到文件
        with open("exam_points_hierarchy.json", "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, ensure_ascii=False, indent=2)
        
        print("考點層級結構已保存到 exam_points_hierarchy.json")
        
    except Exception as e:
        print(f"處理考點時出錯：{str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # 處理測試
    test_exam_point = "民法第167條代理權係以法律行為授與者、純粹經濟上損失之侵權責任、未成年人侵權行為之法定代理人及僱用人之不真正連帶債務"
    result = split_exam_point(test_exam_point)
    print(f"拆分結果: {result}")
    
    # 處理並更新向量庫
    process_and_update_vector_store() 