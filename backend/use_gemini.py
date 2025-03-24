# use_gemini.py
import os
import json
import random
import re
import tempfile
from dotenv import load_dotenv
from typing import List, Dict
import google.generativeai as genai
import io
import traceback

# 載入 .env 檔案中的環境變數
load_dotenv()

# 設定 Google API 金鑰 - 尝试多个可能的环境变數名
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("請在 .env 檔案中設定 GEMINI_API_KEY 或 GOOGLE_API_KEY")

# 打印API密鑰的前5个字符和長度，用于调试
print(f"API密鑰前5个字符: {api_key[:5] if api_key else 'None'}, 長度: {len(api_key) if api_key else 0}")

try:
    genai.configure(api_key=api_key)
    print("Gemini API配置成功")
except Exception as e:
    print(f"Gemini API配置失败: {str(e)}")
    traceback.print_exc()

def extract_text_from_pdf(pdf_bytes, password=None):
    """
    从PDF字节数据中提取文字
    
    Args:
        pdf_bytes: PDF文件的字节数据
        password: PDF文件的密碼，如果有的話
    """
    try:
        # 嘗試導入PyPDF2
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            print("缺少PyPDF2函式庫，正在嘗試安裝...")
            import subprocess
            subprocess.check_call(["pip", "install", "PyPDF2"])
            from PyPDF2 import PdfReader
        
        # 嘗試導入pycryptodome
        try:
            import Crypto
        except ImportError:
            print("缺少pycryptodome函式庫，正在嘗試安裝...")
            import subprocess
            subprocess.check_call(["pip", "install", "pycryptodome"])
        
        # 使用 PyPDF2 讀取 PDF
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PdfReader(pdf_file)
        
        # 檢查PDF是否加密
        if pdf_reader.is_encrypted:
            if password:
                try:
                    # 嘗試使用提供的密碼解密
                    pdf_reader.decrypt(password)
                    print("成功使用提供的密碼解密PDF")
                except Exception as e:
                    return None, f"PDF檔案已加密，提供的密碼無效: {str(e)}"
            else:
                # 嘗試使用空密碼解密（有些PDF只是標記為加密但沒有實際密碼）
                try:
                    pdf_reader.decrypt('')
                    print("成功使用空密碼解密PDF")
                except:
                    return None, "PDF檔案已加密，無法提取文字。請提供未加密的PDF檔案或提供密碼。"
        
        # 提取所有頁面的文字
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        if not text.strip():
            print("PyPDF2未能提取到文字，嘗試使用pdfplumber...")
            return extract_text_with_pdfplumber(pdf_bytes, password)
        
        return text, None
    except Exception as e:
        error_message = f"使用PyPDF2提取PDF文字時發生錯誤: {str(e)}"
        print(error_message)
        traceback.print_exc()
        
        # 嘗試使用備用方法
        print("嘗試使用pdfplumber作為備用方法...")
        return extract_text_with_pdfplumber(pdf_bytes, password)

def extract_text_with_pdfplumber(pdf_bytes, password=None):
    """使用pdfplumber提取PDF文字（作為備用方法）"""
    try:
        # 嘗試導入pdfplumber
        try:
            import pdfplumber
        except ImportError:
            print("缺少pdfplumber函式庫，正在嘗試安裝...")
            import subprocess
            subprocess.check_call(["pip", "install", "pdfplumber"])
            import pdfplumber
        
        # 使用pdfplumber讀取PDF
        pdf_file = io.BytesIO(pdf_bytes)
        
        # 如果有密碼，使用密碼開啟
        if password:
            pdf = pdfplumber.open(pdf_file, password=password)
        else:
            try:
                pdf = pdfplumber.open(pdf_file)
            except:
                return None, "PDF檔案可能已加密，無法提取文字。請提供密碼。"
        
        # 提取所有頁面的文字
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        pdf.close()
        
        if not text.strip():
            return None, "PDF檔案中沒有可提取的文字。可能是掃描件或圖片PDF。"
        
        return text, None
    except Exception as e:
        error_message = f"使用pdfplumber提取PDF文字時發生錯誤: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return None, error_message

def process_pdf_with_gemini(pdf_bytes, filename, password=None):
    """
    使用 Gemini API 處理 PDF 檔案，提取題目並結構化
    
    Args:
        pdf_bytes: PDF檔案的位元組資料
        filename: PDF檔案名
        password: PDF檔案的密碼，如果有的話
    """
    try:
        # 將 PDF 轉換為文字
        pdf_text, error = extract_text_from_pdf(pdf_bytes, password)
        if not pdf_text:
            return {"error": error or "無法從PDF中提取文字"}
        
        print(f"成功從PDF提取文字，長度: {len(pdf_text)} 字元")
        
        # 使用 Gemini API 處理文字
        try:
            print("正在建立Gemini模型實例...")
            model = genai.GenerativeModel('gemini-2.0-flash')
            print("Gemini模型實例建立成功")
            
            # 將文字分成多個區塊進行處理
            text_blocks = []
            block_size = 8000  # 每區塊大約8000字元
            total_length = len(pdf_text)
            
            for i in range(0, total_length, block_size):
                block = pdf_text[i:i + block_size]
                text_blocks.append(block)
            
            print(f"文字已分成 {len(text_blocks)} 個區塊進行處理")
            
            all_questions = []
            current_id = 1
            
            for block_num, block in enumerate(text_blocks, 1):
                print(f"處理第 {block_num}/{len(text_blocks)} 個文字區塊...")
                
                prompt = f"""
                分析以下考試題目文字，提取題目資訊並以JSON格式返回。從題號 {current_id} 開始編號。

                格式要求：
                1. 使用簡單的JSON格式
                2. 選項使用鍵值對，如 "A": "選項內容"
                3. 答案只寫選項字母
                4. 多個答案用逗號分隔
                
                JSON格式示例：
                {{
                  "questions": [
                    {{
                      "id": {current_id},
                      "content": "題幹",
                      "options": {{"A": "選項A", "B": "選項B"}},
                      "answer": "A",
                      "explanation": "解析"
                    }}
                  ]
                }}

                考試題目文字：
                {block}

                只返回JSON格式資料，不要有其他文字。確保JSON格式完全正確。
                """
                
                try:
                    response = model.generate_content(prompt)
                    result_text = response.text
                    
                    # 清理和規範化JSON文字
                    def clean_json_text(text):
                        # 移除可能的 Markdown 程式碼區塊標記
                        text = re.sub(r'```(?:json)?\s*|```', '', text)
                        # 修正非法的轉義序列：用正則表達式匹配不屬於合法轉義序列的反斜線，並替換為雙反斜線
                        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
                        # 去除前後空白符
                        text = text.strip()
                        # 提取最外層的花括號及其內容
                        match = re.search(r'({.*})', text, re.DOTALL)
                        if match:
                            return match.group(1)
                        return text

                    # 清理並解析JSON
                    cleaned_text = clean_json_text(result_text)
                    try:
                        block_result = json.loads(cleaned_text)
                    except json.JSONDecodeError as je:
                        print(f"JSON解析錯誤: {je}")
                        print(f"嘗試解析的文字: {cleaned_text}")
                        # 嘗試使用eval作為備用方案（僅用於除錯）
                        try:
                            import ast
                            block_result = ast.literal_eval(cleaned_text)
                        except:
                            print("備用解析方法也失敗")
                            continue
                    
                    if "questions" in block_result and block_result["questions"]:
                        block_questions = block_result["questions"]
                        
                        # 驗證和修復每個題目的資料
                        for question in block_questions:
                            # 確保所有必要欄位都存在
                            if "id" not in question:
                                question["id"] = current_id
                                current_id += 1
                            
                            if "content" not in question or not question["content"]:
                                print(f"題目 {question.get('id', '未知')} 缺少內容，跳過")
                                continue
                            
                            if "options" not in question or not isinstance(question["options"], dict):
                                question["options"] = {}
                            
                            # 確保answer欄位存在且不為None
                            if "answer" not in question or question["answer"] is None:
                                question["answer"] = ""
                            
                            # 確保explanation欄位存在且不為None
                            if "explanation" not in question or question["explanation"] is None:
                                question["explanation"] = ""
                            
                            # 將None值轉換為空字串
                            question["answer"] = question["answer"] if question["answer"] is not None else ""
                            question["explanation"] = question["explanation"] if question["explanation"] is not None else ""
                        
                        # 過濾掉無效的題目
                        valid_questions = [q for q in block_questions if q.get("content")]
                        all_questions.extend(valid_questions)
                        current_id = max((q["id"] for q in valid_questions), default=current_id) + 1
                        print(f"從第 {block_num} 個區塊中提取了 {len(valid_questions)} 道題目")
                    
                except Exception as block_error:
                    print(f"處理第 {block_num} 個區塊時出錯: {str(block_error)}")
                    continue
            
            if all_questions:
                print(f"總共成功提取 {len(all_questions)} 道題目")
                return {"questions": all_questions}
            else:
                print("沒有成功提取任何題目，嘗試使用備用方法...")
                raise Exception("主要處理方法未能提取題目")
        
        except Exception as api_error:
            print(f"呼叫Gemini API時發生錯誤: {str(api_error)}")
            traceback.print_exc()
            return {"error": f"呼叫Gemini API時發生錯誤: {str(api_error)}"}
    
    except Exception as e:
        error_message = f"處理 PDF 時發生錯誤: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return {"error": error_message}
def adapt_questions(questions: List[Dict]) -> List[Dict]:
    """
    使用 Gemini API 改編原始題目，生成全新多選題：
      - 重新表述題目內容，前綴【改編】
      - 生成4個選項（A、B、C、D）
      - 產生正確答案與詳細解析
      - 回傳格式為 JSON，符合前端需求
    此處對每個題目都重新分配唯一的題號，並在來源中說明原始考試與題目編號。
    """
    adapted = []
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        print(f"建立 Gemini 模型失敗: {str(e)}")
        model = None  # 將使用預設邏輯

    for index, q in enumerate(questions):
        # 取得原始metadata，並定義題號與來源資訊
        metadata = q.get("metadata", {})
        question_number = metadata.get("question_number", index + 1)
        original_exam_name = metadata.get("exam_name", "未知考試")
        source_str = f"改編自 {original_exam_name} 試題第 {question_number} 題"
        # 重新分配唯一題號（依照循環順序）
        new_id = index + 1

        # 建立 Gemini prompt：
        prompt = f"""
請根據以下原始題目信息，生成一份全新的多選題，請以JSON格式返回，格式要求如下：
{{
  "id": {new_id},
  "content": "【改編】重新表述的題目內容",
  "options": {{"A": "選項A", "B": "選項B", "C": "選項C", "D": "選項D"}},
  "answer": "正確選項字母",
  "explanation": "【改編】詳細解析",
  "source": "{source_str}"
}}

原始題目內容: {q.get("content", "")}
原始選項: {json.dumps(q.get("options", {}), ensure_ascii=False)}
原始答案: {metadata.get("answer", "")}
原始解析: {metadata.get("explanation", "")}
只返回符合上述格式的純JSON，無其他文字。
"""
        adapted_q = None
        if model:
            try:
                response = model.generate_content(prompt)
                result_text = response.text
                # 移除可能的 markdown code block
                import re
                def clean_json_text(text):
                    text = re.sub(r'```(?:json)?\s*|```', '', text).strip()
                    match = re.search(r'({.*})', text, re.DOTALL)
                    return match.group(1) if match else text
                cleaned_text = clean_json_text(result_text)
                adapted_q = json.loads(cleaned_text)
            except Exception as e:
                print(f"改編題目失敗: {str(e)}")
        if not adapted_q:
            # 使用預設邏輯
            adapted_q = {
                "id": new_id,
                "content": "【改編】" + q.get("content", ""),
                "options": q.get("options") if q.get("options") and isinstance(q.get("options"), dict) and len(q.get("options")) >= 2 
                           else {"A": "選項 A", "B": "選項 B", "C": "選項 C", "D": "選項 D"},
                "answer": metadata.get("answer", ""),
                "explanation": "【改編】" + metadata.get("explanation", ""),
                "source": source_str
            }
        # 確保選項有四個
        if not adapted_q.get("options") or len(adapted_q["options"]) != 4:
            adapted_q["options"] = {"A": "選項 A", "B": "選項 B", "C": "選項 C", "D": "選項 D"}
        adapted.append(adapted_q)
    return adapted

def retrieve_online_questions(keyword: str, num_questions: int) -> List[Dict]:
    """
    當內部題庫中未找到相關題目時，
    透過 Gemini API 生成與指定關鍵字相關的全新考試多選題，
    每題須包含題號、題幹、4個選項、正確答案（選項字母）與詳細解析，
    並要求返回純 JSON 格式的數據。
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        print(f"建立 Gemini 模型失敗: {str(e)}")
        return []
    
    prompt = f"""
請根據網路上的公開資源，生成 {num_questions} 道關於 "{keyword}" 的全新多選題，
每題包括題號、題幹、4個選項（鍵值對形式，分別為 "A", "B", "C", "D"）、正確答案（僅為選項字母）和詳細解析。
請以以下 JSON 格式返回，僅返回符合格式的純 JSON，不包含其他文字：
{{
  "questions": [
    {{
      "id": 1,
      "content": "題幹內容",
      "options": {{"A": "選項A", "B": "選項B", "C": "選項C", "D": "選項D"}},
      "answer": "A",
      "explanation": "解析內容"
    }}
  ]
}}
"""
    try:
        response = model.generate_content(prompt)
        result_text = response.text
        # 清除 Markdown code block 標記，提取 JSON 部分
        import re, json
        def clean_json_text(text):
            text = re.sub(r'```(?:json)?\s*|```', '', text).strip()
            match = re.search(r'({.*})', text, re.DOTALL)
            return match.group(1) if match else text
        cleaned_text = clean_json_text(result_text)
        online_result = json.loads(cleaned_text)
        if "questions" in online_result:
            return online_result["questions"]
        else:
            return []
    except Exception as e:
        print(f"在線檢索題目失敗: {str(e)}")
        return []

if __name__ == "__main__":
    # 測試用範例
    with open("sample.pdf", "rb") as f:
        pdf_bytes = f.read()
    result = process_pdf_with_gemini(pdf_bytes, "sample.pdf")
    print(json.dumps(result, ensure_ascii=False, indent=2))