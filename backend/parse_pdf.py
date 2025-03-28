# parse_pdf.py
import os
import re
import json
from typing import List, Dict, Any, Tuple
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# 設定 pytesseract 路徑 (Windows 環境可能需要)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_pdf(pdf_bytes: bytes, password=None) -> str:
    """將 PDF 轉換為文字"""
    try:
        # 將 PDF 轉換為圖像，如果有密碼則提供密碼
        images = convert_from_bytes(pdf_bytes, userpw=password) if password else convert_from_bytes(pdf_bytes)
        
        # OCR 處理每個頁面並合併文字
        full_text = ""
        for img in images:
            # 使用繁體中文語言包進行 OCR
            text = pytesseract.image_to_string(img, lang='chi_tra+eng')
            full_text += text + "\n\n"
        
        return full_text
    except Exception as e:
        if "password" in str(e).lower() or "加密" in str(e):
            if password:
                raise Exception(f"提供的密碼無法解密PDF：{str(e)}")
            else:
                raise Exception("PDF檔案已加密，請提供密碼")
        raise Exception(f"處理PDF時發生錯誤：{str(e)}")

def parse_questions(text: str, exam_name: str) -> List[Dict[str, Any]]:
    """解析文字提取題目資料"""
    # 分離所有問題 (假設題目格式為 "1. 問題內容")
    question_pattern = r'(\d+)[\.、]\s+(.*?)(?=\d+[\.、]|\Z)'
    raw_questions = re.findall(question_pattern, text, re.DOTALL)
    
    parsed_questions = []
    
    for question_num, content in raw_questions:
        try:
            # 清理並分析題目文字
            processed_content = content.strip()
            
            # 提取選項 (A、B、C、D)
            options_pattern = r'([A-D])[\s、.]+([^\n]+)'
            options_matches = re.findall(options_pattern, processed_content)
            
            # 如果找不到選項格式，可能是題目格式不符
            if not options_matches or len(options_matches) < 4:
                continue
                
            # 提取題目內容 (選項前的文字)
            first_option_pos = processed_content.find(options_matches[0][0])
            question_content = processed_content[:first_option_pos].strip()
            
            # 構建選項字典
            options = {}
            for opt_letter, opt_content in options_matches:
                options[opt_letter] = opt_content.strip()
            
            # 尋找答案 (假設答案格式為 "答案：X" 或 "正確答案：X")
            answer_match = re.search(r'答案[：:]\s*([A-D])', processed_content)
            answer = answer_match.group(1) if answer_match else ""
            
            # 尋找解析 (假設解析格式為 "解析：..." 或 "解釋：...")
            explanation_match = re.search(r'解[析釋說][：:]\s*(.*?)(?=\d+[\.、]|\Z)', processed_content, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else ""
            
            # 構建問題字典
            question_dict = {
                "id": int(question_num),
                "content": question_content,
                "options": options,
                "answer": answer,
                "explanation": explanation
            }
            
            parsed_questions.append(question_dict)
        except Exception as e:
            print(f"解析第 {question_num} 題時出錯: {str(e)}")
            continue
    
    return parsed_questions

def save_to_json(questions: List[Dict[str, Any]], exam_name: str, output_dir: str = "downloads") -> str:
    """將解析後的問題保存為 JSON 檔案"""
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 預處理問題，確保所有值都是可序列化的，並檢查ID唯一性
    processed_questions = []
    seen_ids = set()
    
    try:
        # 嘗試導入增強的JSON處理模塊
        from process_json import EnhancedJSONEncoder, clean_for_json
        use_enhanced_encoder = True
        print("使用增強的JSON處理模塊")
    except ImportError:
        # 如果模塊不可用，使用本地定義的編碼器
        use_enhanced_encoder = False
        print("使用本地JSON編碼器")
        
        # 自定義JSON編碼器，處理NumPy數組和其他特殊類型
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                import sys
                
                # 處理None值
                if obj is None:
                    return ""
                    
                # 處理NumPy數組
                if 'numpy' in sys.modules:
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                        
                    # 處理NumPy標量
                    if isinstance(obj, (np.integer, np.floating, np.bool_)):
                        return obj.item()
                        
                    # 處理可能導致"The truth value of an array"錯誤的對象
                    if hasattr(obj, 'any') and hasattr(obj, 'all') and hasattr(obj, 'shape'):
                        # 是類數組對象，轉換為列表
                        try:
                            return np.asarray(obj).tolist()
                        except:
                            pass
                
                # 處理集合類型
                if isinstance(obj, set):
                    return list(obj)
                    
                # 處理元組
                if isinstance(obj, tuple):
                    return list(obj)
                    
                # 處理日期和時間
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                    
                # 處理可轉換的對象
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                    
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                    
                # 處理其他不可序列化的類型
                try:
                    return super().default(obj)
                except TypeError:
                    # 如果無法序列化，轉換為字符串
                    return str(obj)
    
    for index, question in enumerate(questions):
        # 檢查是否有重複ID
        q_id = question.get("id")
        if q_id in seen_ids:
            # 如果ID重複，添加後綴區分
            q_id = f"{q_id}_{index}" 
            question["id"] = q_id
        
        seen_ids.add(q_id)
        
        processed_question = {}
        for key, value in question.items():
            if key == "exam_point" and value is None:
                processed_question[key] = ""
            elif key in ["keywords", "law_references"] and value is None:
                processed_question[key] = []
            else:
                processed_question[key] = value
        
        processed_questions.append(processed_question)
    
    # 創建 JSON 結構
    output_data = {
        "exam": exam_name,
        "questions": processed_questions
    }
    
    # 生成檔案名
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"questions_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 寫入 JSON 檔案，使用自定義編碼器
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            if use_enhanced_encoder:
                # 如果可用，使用增強的JSON處理
                # 先清理數據以確保可序列化
                output_data = clean_for_json(output_data)
                json.dump(output_data, f, ensure_ascii=False, indent=2, cls=EnhancedJSONEncoder)
            else:
                # 否則使用本地編碼器
                json.dump(output_data, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
        print(f"成功保存JSON文件: {filename}")
    except Exception as e:
        print(f"JSON序列化錯誤: {str(e)}")
        
        # 嘗試使用更安全的序列化方法
        try:
            if use_enhanced_encoder:
                from process_json import safe_json_dumps
                # 使用safe_json_dumps來序列化
                json_str = safe_json_dumps(output_data, ensure_ascii=False, indent=2)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(json_str)
                print(f"使用安全序列化方法保存: {filename}")
                return filename
        except Exception:
            pass
            
        # 如果上述方法都失敗，使用最保守的方法
        safe_output = {
            "exam": exam_name,
            "questions": [],
            "error": f"原始序列化失敗: {str(e)}"
        }
        
        # 安全的保存方式
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(safe_output, f, ensure_ascii=False, indent=2)
        print(f"使用最保守方法保存: {filename}")
    
    return filename

def process_pdf_with_gemini(pdf_bytes, filename, password=None):
    """
    處理PDF檔案並使用Gemini API解析結構化資料
    """
    try:
        # 使用extract_text_from_pdf提取文字
        extracted_text = extract_text_from_pdf(pdf_bytes, password)
        
        if not extracted_text or len(extracted_text) < 50:
            return {"error": "無法從PDF中提取足夠的文字內容，請確認PDF檔案是否包含可識別的文字"}
        
        print(f"成功從PDF提取文字，長度: {len(extracted_text)} 字元")
        
        # 使用API解析文字為結構化資料
        try:
            # 根據當前設定使用合適的API
            if config.USE_GEMINI_API:
                from use_gemini import process_text_with_gemini
                result = process_text_with_gemini(extracted_text, filename)
            else:
                from use_openai import process_text_with_openai
                result = process_text_with_openai(extracted_text, filename)
                
            # 檢查是否有警告信息
            if isinstance(result, dict) and "warning" in result:
                warning_message = result["warning"]
                questions = result.get("questions", [])
                
                if len(questions) > 0:
                    # 有部分成功結果
                    return {
                        "questions": questions,
                        "warning": warning_message,
                        "quota_exceeded": "quota" in warning_message.lower() or "配額" in warning_message
                    }
                else:
                    # 完全失敗
                    return {"error": warning_message}
                
            # 正常情況，返回完整結果
            return {"questions": result}
            
        except Exception as api_error:
            error_message = str(api_error)
            print(f"API處理文字時出錯: {error_message}")
            
            # 檢查是否是配額限制錯誤
            if "429" in error_message or "quota" in error_message.lower() or "配額" in error_message:
                # 如果API配額已超出，嘗試切換到另一個API
                print("檢測到API配額限制，嘗試切換到另一個API...")
                
                # 切換API
                config.USE_GEMINI_API = not config.USE_GEMINI_API
                
                try:
                    # 使用另一個API重試
                    if config.USE_GEMINI_API:
                        from use_gemini import process_text_with_gemini
                        result = process_text_with_gemini(extracted_text, filename)
                    else:
                        from use_openai import process_text_with_openai
                        result = process_text_with_openai(extracted_text, filename)
                    
                    # 如果返回的是帶有警告的部分結果
                    if isinstance(result, dict) and "warning" in result:
                        return {
                            "questions": result.get("questions", []),
                            "warning": f"已切換到{'Gemini' if config.USE_GEMINI_API else 'OpenAI'} API，但是: {result['warning']}",
                            "quota_exceeded": True
                        }
                    
                    # 如果成功切換且完全處理
                    return {
                        "questions": result,
                        "warning": f"已切換到{'Gemini' if config.USE_GEMINI_API else 'OpenAI'} API處理"
                    }
                    
                except Exception as retry_error:
                    error_msg = str(retry_error)
                    print(f"切換API後仍然失敗: {error_msg}")
                    
                    # 檢查結果中是否包含部分成功的結果
                    if hasattr(retry_error, 'partial_results') and retry_error.partial_results:
                        return {
                            "questions": retry_error.partial_results,
                            "error": f"API處理失敗: {error_msg}",
                            "warning": "返回部分處理的結果，可能不完整",
                            "quota_exceeded": True
                        }
                    
                    return {"error": f"所有API都遇到配額限制或處理錯誤: {error_msg}"}
            
            # 如果API錯誤帶有部分結果
            if hasattr(api_error, 'partial_results') and api_error.partial_results:
                return {
                    "questions": api_error.partial_results,
                    "error": f"API處理失敗: {error_message}",
                    "warning": "返回部分處理的結果，可能不完整"
                }
            
            return {"error": f"API處理失敗: {error_message}"}
    
    except Exception as e:
        error_message = str(e)
        print(f"處理PDF時發生錯誤: {error_message}")
        
        # 檢查是否是加密PDF錯誤
        if "加密" in error_message or "password" in error_message.lower():
            if not password:
                return {"error": "PDF檔案已加密，請提供密碼"}
            else:
                return {"error": "無法使用提供的密碼解密PDF，請確認密碼是否正確"}
        
        return {"error": f"處理PDF時發生錯誤: {error_message}"}