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

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """將 PDF 轉換為文字"""
    # 將 PDF 轉換為圖像
    images = convert_from_bytes(pdf_bytes)
    
    # OCR 處理每個頁面並合併文字
    full_text = ""
    for img in images:
        # 使用繁體中文語言包進行 OCR
        text = pytesseract.image_to_string(img, lang='chi_tra+eng')
        full_text += text + "\n\n"
    
    return full_text

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
    
    # 創建 JSON 結構
    output_data = {
        "exam": exam_name,
        "questions": questions
    }
    
    # 生成檔案名
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"questions_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 寫入 JSON 檔案
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return filename