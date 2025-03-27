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
import time
import requests
from google.api_core.exceptions import ResourceExhausted
import logging

# 載入 .env 檔案中的環境變數
load_dotenv()

# 全局變量，控制當前上傳過程中使用的API
# 默認使用Gemini API
USE_OPENAI_FOR_CURRENT_BATCH = False

# 設定 Google API 金鑰 - 尝试多个可能的环境變數名
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

# 設置全局日誌級別為WARNING或ERROR，減少INFO級別的輸出
logging.basicConfig(level=logging.WARNING)

# 或者針對特定模塊設置日誌級別
logging.getLogger("use_gemini").setLevel(logging.WARNING)

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

def extract_legal_metadata(question_content, explanation=""):
    """
    提取題目的法律元數據：考點、關鍵字和法條引用
    
    Args:
        question_content: 題目內容
        explanation: 題目解析（如果有）
    
    Returns:
        包含考點、關鍵字和法條引用的字典
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = """
        你是一位專業的法律考試分析專家，精通台灣法律考試的分析與整理。請分析以下法律考題及其解析，提取具體考點、關鍵字和法條引用：
        
        【題目】
        {question}
        
        【解析】
        {explanation}
        
        請針對此題目提取以下三項元素：
        
        1. 【考點】：提取一個具體明確的法律考點（例如「物權行為無因性」、「民法代理權授與」、「消滅時效中斷事由」），必須使用精確的法律術語，避免過於廣泛的分類（如「民法」或「刑法」）。考點應該是該題目測試的核心法律概念或原則。
        
        2. 【關鍵字】：列出3-5個與題目直接相關的法律關鍵詞，應包含:
           - 核心法律概念（如「無權代理」、「債權讓與」）
           - 法律行為類型（如「買賣契約」、「贈與」）
           - 法律效果（如「返還請求權」、「撤銷權」）
           - 例外或限制條件（如「善意第三人」、「特殊時效期間」）
           
        3. 【法條引用】：精確列出題目涉及的法條（如「民法第184條」、「公司法第27條」），包括主要條文及相關條文。如果題目或解析中明確提及特定法條，必須包含；如果沒有明確提及但有明顯相關法條，也請列出。格式統一為「法律名稱+第X條」。
        
        請以JSON格式返回，確保格式正確：
        ```json
        {{
          "exam_point": "具體法律考點",
          "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3", "關鍵詞4", "關鍵詞5"],
          "law_references": ["民法第X條", "民法第Y條"]
        }}
        ```
        
        重要提示：
        - 每個關鍵詞應該是具體的法律概念，而非一般性描述
        - 考點必須是一個明確的法律原則或規則，不可過於籠統
        - 如果無法確定某項內容，對應欄位請返回空值（考點為空字串，關鍵詞或法條為空陣列）
        - 只返回JSON格式，不要有其他說明文字
        """.format(question=question_content, explanation=explanation)
        
        response = model.generate_content(prompt)
        response_text = response.text
        
        # 清理回應文本，確保只包含JSON部分
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text.strip()
        
        try:
            import json
            result = json.loads(json_text)
            
            # 確保結果包含所有必要的字段
            if "exam_point" not in result:
                result["exam_point"] = ""
            if "keywords" not in result:
                result["keywords"] = []
            if "law_references" not in result:
                result["law_references"] = []
                
            return result
        except json.JSONDecodeError as e:
            print(f"無法解析JSON: {str(e)}, 回應文本: {response_text}")
            return {"exam_point": "", "keywords": [], "law_references": []}
            
    except Exception as e:
        print(f"提取法律元數據時發生錯誤: {str(e)}")
        # 降級機制 - 使用簡單規則產生基本元數據
        keywords = []
        law_references = []
        exam_point = ""
        
        # 簡單關鍵詞提取
        if "代理" in question_content:
            keywords.append("代理")
            exam_point = "代理權"
        if "民法第" in explanation:
            # 嘗試提取法條引用
            import re
            law_refs = re.findall(r'民法第 *\d+ *條', explanation)
            law_references.extend(law_refs)
            
        return {
            "exam_point": exam_point,
            "keywords": keywords,
            "law_references": law_references
        }

def extract_legal_metadata_with_retry(question_content, explanation="", max_retries=3, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            # 原有的 extract_legal_metadata 代碼...
            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = """
            你是一位專業的法律考試分析專家，精通台灣法律考試的分析與整理。請分析以下法律考題及其解析，提取具體考點、關鍵字和法條引用：
            
            【題目】
            {question}
            
            【解析】
            {explanation}
            
            請針對此題目提取以下三項元素：
            
            1. 【考點】：提取一個具體明確的法律考點（例如「物權行為無因性」、「民法代理權授與」、「消滅時效中斷事由」），必須使用精確的法律術語，避免過於廣泛的分類（如「民法」或「刑法」）。考點應該是該題目測試的核心法律概念或原則。
            
            2. 【關鍵字】：列出3-5個與題目直接相關的法律關鍵詞，應包含:
               - 核心法律概念（如「無權代理」、「債權讓與」）
               - 法律行為類型（如「買賣契約」、「贈與」）
               - 法律效果（如「返還請求權」、「撤銷權」）
               - 例外或限制條件（如「善意第三人」、「特殊時效期間」）
               
            3. 【法條引用】：精確列出題目涉及的法條（如「民法第184條」、「公司法第27條」），包括主要條文及相關條文。如果題目或解析中明確提及特定法條，必須包含；如果沒有明確提及但有明顯相關法條，也請列出。格式統一為「法律名稱+第X條」。
            
            請以JSON格式返回，確保格式正確：
            ```json
            {{
              "exam_point": "具體法律考點",
              "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3", "關鍵詞4", "關鍵詞5"],
              "law_references": ["民法第X條", "民法第Y條"]
            }}
            ```
            
            重要提示：
            - 每個關鍵詞應該是具體的法律概念，而非一般性描述
            - 考點必須是一個明確的法律原則或規則，不可過於籠統
            - 如果無法確定某項內容，對應欄位請返回空值（考點為空字串，關鍵詞或法條為空陣列）
            - 只返回JSON格式，不要有其他說明文字
            """.format(question=question_content, explanation=explanation)
            
            response = model.generate_content(prompt)
            response_text = response.text
            
            # 清理回應文本，確保只包含JSON部分
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()
            else:
                json_text = response_text.strip()
            
            try:
                import json
                result = json.loads(json_text)
                
                # 確保結果包含所有必要的字段
                if "exam_point" not in result:
                    result["exam_point"] = ""
                if "keywords" not in result:
                    result["keywords"] = []
                if "law_references" not in result:
                    result["law_references"] = []
                    
                return result
            except json.JSONDecodeError as e:
                print(f"無法解析JSON: {str(e)}, 回應文本: {response_text}")
                return {"exam_point": "", "keywords": [], "law_references": []}
        except Exception as e:
            # 如果是配額錯誤，等待一段時間後重試
            if "429" in str(e):
                wait_time = backoff_factor ** attempt
                print(f"API配額限制，等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
                continue
            else:
                # 其他錯誤使用降級機制處理
                print(f"提取法律元數據時發生錯誤: {str(e)}")
                # 使用降級邏輯...
                keywords = []
                law_references = []
                exam_point = ""
                
                # 簡單關鍵詞提取
                if "代理" in question_content:
                    keywords.append("代理")
                    exam_point = "代理權"
                if "民法第" in explanation:
                    # 嘗試提取法條引用
                    import re
                    law_refs = re.findall(r'民法第 *\d+ *條', explanation)
                    law_references.extend(law_refs)
                    
                return {
                    "exam_point": exam_point,
                    "keywords": keywords,
                    "law_references": law_references
                }
    
    # 如果所有重試都失敗，使用降級機制
    return {"exam_point": "", "keywords": [], "law_references": []}

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
            quota_exceeded = False
            
            for block_num, block in enumerate(text_blocks, 1):
                print(f"處理第 {block_num}/{len(text_blocks)} 個文字區塊...")
                
                # 如果配額已經超限，跳過後續區塊處理
                if quota_exceeded:
                    print(f"跳過區塊 {block_num} 處理，因為API配額已超限")
                    continue
                
                # 確保block不為None
                if block is None:
                    print(f"區塊 {block_num} 內容為空，跳過處理")
                    continue
                
                prompt = f"""
                你是一位專精於台灣法律考試的人工智能，善於從考試資料中精確識別和結構化法律題目。
                請從下面的文本中提取所有法律考試題目，並以JSON格式返回。從題號 {current_id} 開始編號。

                【任務說明】
                1. 仔細分析文本，識別所有獨立的考試題目
                2. 對每個題目進行分類（單選題、多選題、簡答題、申論題等）
                3. 提取題目內容、選項、答案和解析（如有）
                4. 返回結構化的JSON數據

                【格式要求】
                ```json
                {{
                  "questions": [
                    {{
                      "id": {current_id},
                      "type": "單選題", // 可能的值: 單選題、多選題、簡答題、申論題、是非題
                      "content": "完整題幹內容，包含題號和題目描述",
                      "options": {{"A": "選項A內容", "B": "選項B內容", "C": "選項C內容", "D": "選項D內容"}},
                      "answer": "A", // 單選題填寫選項字母，多選題用逗號分隔，如"A,C"
                      "explanation": "解析內容（如文本中提供）"
                    }}
                    // ... 更多題目
                  ]
                }}
                ```

                【特別注意】
                1. 題型判斷:
                   - 單選題通常指示「下列何者」或明確表明只有一個答案
                   - 多選題通常表述為「下列何者為正確」或明確表明有多個答案
                   - 簡答題通常要求簡短文字回答
                   - 申論題通常需要長篇分析討論
                   - 是非題通常只需判斷對錯

                2. 答案處理:
                   - 單選題：只填寫一個選項字母（如"A"）
                   - 多選題：用逗號分隔多個選項字母（如"A,C,D"）
                   - 其他題型：填入文字答案或保留空字串

                3. 選項格式:
                   - 確保選項使用正確的鍵值對格式
                   - 如果是簡答題或申論題等沒有選項的題型，options應為空對象 {{}}

                請分析以下考試文本:
                {block}
                
                僅返回有效的JSON格式，不要加入其他說明文字。如果無法識別任何完整題目，返回空的questions陣列。
                """
                
                try:
                    # 嘗試使用Gemini API處理
                    max_retries = 2
                    retry_count = 0
                    backoff_time = 1
                    
                    while retry_count <= max_retries:
                        try:
                            response = model.generate_content(prompt)
                            response_text = response.text
                            break  # 如果成功，退出重試循環
                        except Exception as retry_error:
                            retry_count += 1
                            # 檢查是否是配額錯誤
                            if "429" in str(retry_error) or "quota" in str(retry_error).lower():
                                print(f"API配額限制錯誤，嘗試第 {retry_count}/{max_retries} 次重試...")
                                # 最後一次重試仍然失敗
                                if retry_count > max_retries:
                                    quota_exceeded = True
                                    print(f"處理區塊 {block_num} 時出錯: {str(retry_error)}")
                                    break
                                # 指數退避
                                import time
                                time.sleep(backoff_time)
                                backoff_time *= 2
                            else:
                                # 其他錯誤直接拋出
                                raise
                    
                    # 如果配額超限，跳出處理
                    if quota_exceeded:
                        print("配額限制已達，停止處理後續區塊")
                        break
                    
                    # 清理回應文本，確保只包含JSON部分
                    if "```json" in response_text:
                        json_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        json_text = response_text.split("```")[1].strip()
                    else:
                        json_text = response_text.strip()
                    
                    try:
                        result = json.loads(json_text)
                        if "questions" in result:
                            block_questions = result["questions"]
                            print(f"成功從區塊 {block_num} 提取出 {len(block_questions)} 道題目")
                            
                            # 驗證每個問題的合法性
                            valid_questions = []
                            for q in block_questions:
                                if "content" in q and q["content"] and isinstance(q["content"], str):
                                    # 確保options不為None
                                    if q.get("options") is None:
                                        q["options"] = {}
                                    valid_questions.append(q)
                                else:
                                    print(f"跳過無效題目: {q}")
                            
                            # 使用批處理提取法律元數據
                            if valid_questions:
                                processed_questions = process_questions_in_batches(valid_questions)
                                
                                # 更新當前ID
                                for question in processed_questions:
                                    if question["id"] >= current_id:
                                        current_id = question["id"] + 1
                                
                                all_questions.extend(processed_questions)
                            
                        else:
                            print(f"區塊 {block_num} 解析結果中沒有找到questions字段")
                    except json.JSONDecodeError as e:
                        print(f"無法解析JSON: {str(e)}, 回應文本: {response_text}")
                        continue
                except Exception as e:
                    print(f"處理區塊 {block_num} 時出錯: {str(e)}")
                    # 檢查是否是配額錯誤
                    if "429" in str(e) or "quota" in str(e).lower():
                        quota_exceeded = True  # 設置配額超限標誌
                    continue
            
            print(f"總共提取出 {len(all_questions)} 道題目")
            
            if not all_questions:
                if quota_exceeded:
                    return {"error": "Gemini API 配額超限，無法完成所有處理", "quota_exceeded": True}
                else:
                    return {"error": "未能從PDF中提取有效題目"}
            
            # 即使配額超限，也返回已處理的題目
            result = {"questions": all_questions}
            if quota_exceeded:
                result["warning"] = "由於API配額限制，僅提取了部分題目"
            
            return result
            
        except Exception as e:
            error_message = f"使用Gemini API處理文字時發生錯誤: {str(e)}"
            print(error_message)
            traceback.print_exc()
            return {"error": error_message}
    except Exception as e:
        error_message = f"處理PDF時發生錯誤: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return {"error": error_message}

def adapt_questions(questions: List[Dict]) -> List[Dict]:
    """
    使用 Gemini API 改編原始題目，生成全新題目：
      - 正確識別並保留題目類型（單選題/多選題）
      - 根據原始題目類型重新生成相應類型的題目
      - 生成正確數量的選項
      - 產生與題型匹配的正確答案與詳細解析
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
        
        # 確定題目類型
        original_type = metadata.get("type", "單選題")
        # 檢查原始答案是否包含逗號，來確認是否為多選題
        original_answer = metadata.get("answer", "")
        if "," in original_answer and original_type != "多選題":
            question_type = "多選題"
            print(f"答案格式顯示題目 #{new_id} 可能是多選題 (答案:{original_answer})，但標記為 {original_type}，已自動更正")
        elif original_type == "多選題" and "," not in original_answer:
            question_type = "單選題"
            print(f"題目 #{new_id} 標記為多選題，但答案 '{original_answer}' 不含逗號，已自動更正為單選題")
        else:
            question_type = original_type

        # 建立 Gemini prompt：
        prompt = f"""
請根據以下原始題目信息，生成一份全新的{question_type}，請以JSON格式返回，格式要求如下：
{{
  "id": {new_id},
  "content": "【改編】重新表述的題目內容",
  "options": {{"A": "選項A", "B": "選項B", "C": "選項C", "D": "選項D"}},
  "answer": "{'正確選項字母(如為多選題，用逗號分隔如A,C)' if question_type == '多選題' else '正確選項字母(單一字母)'}", 
  "explanation": "【改編】詳細解析",
  "source": "{source_str}",
  "type": "{question_type}"
}}

原始題目類型: {question_type}
原始題目內容: {q.get("content", "")}
原始選項: {json.dumps(q.get("options", {}), ensure_ascii=False)}
原始答案: {metadata.get("answer", "")}
原始解析: {metadata.get("explanation", "")}

特別注意：
1. 嚴格符合題目類型的要求，{ "如果是單選題，答案必須是單一字母如'A'；如果是多選題，答案必須用逗號分隔多個選項，如'A,C'" if question_type == '多選題' else "這是單選題，答案必須是單一字母，如'B'" }
2. 所有選項必須使用「A」、「B」、「C」、「D」作為鍵值
3. 只返回符合上述格式的純JSON，無其他文字。
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
                "source": source_str,
                "type": question_type
            }
        
        # 確保選項有四個
        if not adapted_q.get("options") or len(adapted_q["options"]) != 4:
            adapted_q["options"] = {"A": "選項 A", "B": "選項 B", "C": "選項 C", "D": "選項 D"}
        
        # 確保答案格式與題型一致
        answer = adapted_q.get("answer", "")
        if question_type == "單選題" and "," in answer:
            # 單選題答案不應包含逗號，取第一個字母
            adapted_q["answer"] = answer.split(",")[0].strip()
            print(f"修正題目 #{new_id} 的答案格式: 從 '{answer}' 更正為 '{adapted_q['answer']}'")
        elif question_type == "多選題" and "," not in answer and len(answer) > 1:
            # 多選題答案，但沒有用逗號分隔，嘗試修正格式
            adapted_q["answer"] = ",".join(list(answer))
            print(f"修正題目 #{new_id} 的答案格式: 從 '{answer}' 更正為 '{adapted_q['answer']}'")
        
        # 確保題目類型被正確記錄
        adapted_q["type"] = question_type
        
        adapted.append(adapted_q)
    return adapted

def retrieve_online_questions(keyword: str, num_questions: int) -> List[Dict]:
    """
    當內部題庫中未找到相關題目時，
    透過 Gemini API 生成與指定關鍵字相關的全新考試題目，
    生成題目包含單選題和多選題，每題須包含：
    - 題號、題幹、4個選項
    - 正確答案（單選題為單一字母，多選題為逗號分隔的多個字母）
    - 詳細解析和題目類型
    並要求返回純 JSON 格式的數據。
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        print(f"建立 Gemini 模型失敗: {str(e)}")
        return []
    
    prompt = f"""
請根據網路上的公開資源，生成 {num_questions} 道關於 "{keyword}" 的全新題目。
其中一半為單選題，一半為多選題（如總數為奇數，則單選題多一題）。

每題包括：
1. 題號
2. 題幹
3. 4個選項（鍵值對形式，分別為 "A", "B", "C", "D"）
4. 正確答案（單選題為單一字母，多選題為逗號分隔的多個字母）
5. 詳細解析
6. 題目類型（"單選題"或"多選題"）

請以以下 JSON 格式返回，僅返回符合格式的純 JSON，不包含其他文字：
{{
  "questions": [
    {{
      "id": 1,
      "content": "題幹內容",
      "options": {{"A": "選項A", "B": "選項B", "C": "選項C", "D": "選項D"}},
      "answer": "A",
      "explanation": "解析內容",
      "type": "單選題"
    }},
    {{
      "id": 2,
      "content": "題幹內容",
      "options": {{"A": "選項A", "B": "選項B", "C": "選項C", "D": "選項D"}},
      "answer": "A,C",
      "explanation": "解析內容",
      "type": "多選題"
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
        
        # 處理返回的題目，確保題型和答案格式匹配
        if "questions" in online_result:
            questions = online_result["questions"]
            for q in questions:
                # 檢查答案與題型是否匹配
                q_type = q.get("type", "單選題")
                answer = q.get("answer", "")
                
                if q_type == "單選題" and "," in answer:
                    # 單選題答案不應包含逗號，取第一個字母
                    q["answer"] = answer.split(",")[0].strip()
                    print(f"修正從網路檢索的題目 #{q['id']} 的答案格式: 從 '{answer}' 更正為 '{q['answer']}'")
                elif q_type == "多選題" and "," not in answer and len(answer) > 1:
                    # 多選題答案，但沒有用逗號分隔，嘗試修正格式
                    q["answer"] = ",".join(list(answer))
                    print(f"修正從網路檢索的題目 #{q['id']} 的答案格式: 從 '{answer}' 更正為 '{q['answer']}'")
                
                # 確保每題都有四個選項
                if not q.get("options") or len(q["options"]) != 4:
                    q["options"] = {"A": "選項 A", "B": "選項 B", "C": "選項 C", "D": "選項 D"}
                    print(f"修正從網路檢索的題目 #{q['id']} 的選項數量")
            
            return questions
        else:
            return []
    except Exception as e:
        print(f"在線檢索題目失敗: {str(e)}")
        return []

def enhanced_fallback_extraction(question_content, explanation=""):
    """更強大的降級機制，當API失敗時使用"""
    import re
    
    # 預設值
    keywords = []
    law_references = []
    exam_point = ""
    
    # 提取法條引用
    law_pattern = r'民法第\s*\d+\s*條|刑法第\s*\d+\s*條|公司法第\s*\d+\s*條'
    law_refs = re.findall(law_pattern, explanation)
    if law_refs:
        law_references.extend(law_refs)
        if not exam_point and "法" in explanation[:50]:
            # 嘗試從開頭找出可能的考點
            first_sentences = explanation.split('。')[0].split('，')
            for sent in first_sentences:
                if "法" in sent and len(sent) < 30:
                    exam_point = sent.strip()
                    break
    
    # 關鍵詞匹配表（可擴充）
    keyword_map = {
        "代理": ["代理", "代理權", "無權代理"],
        "時效": ["消滅時效", "取得時效"],
        "婚姻": ["結婚", "離婚", "婚姻關係"],
        "繼承": ["繼承", "遺產", "遺囑", "繼承人"],
        "親權": ["親權", "監護權", "扶養"],
        "契約": ["契約", "合同", "債務不履行", "買賣"],
        "物權": ["所有權", "抵押權", "質權", "用益物權"],
        "侵權": ["侵權行為", "損害賠償", "過失責任"],
        "破產": ["破產", "債務清理", "重整"],
        "公司": ["公司", "股東", "董事", "監察人", "股份"]
    }
    
    # 檢查關鍵詞
    for key_term, related_terms in keyword_map.items():
        if key_term in question_content or key_term in explanation:
            for term in related_terms:
                if term in question_content or term in explanation:
                    keywords.append(term)
            if not exam_point and len(related_terms) > 0:
                for term in related_terms:
                    if term in question_content or term in explanation:
                        exam_point = term
                        break
    
    # 提取考點（如果仍未找到）
    if not exam_point and len(question_content) > 10:
        # 嘗試在問題第一句找考點
        first_sentence = question_content.split('，')[0]
        if len(first_sentence) < 20 and "下列" not in first_sentence:
            exam_point = first_sentence
    
    # 去重
    keywords = list(set(keywords))
    law_references = list(set(law_references))
    
    return {
        "exam_point": exam_point,
        "keywords": keywords,
        "law_references": law_references
    }

def extract_legal_metadata_with_openai(question_content, explanation=""):
    """
    使用OpenAI API提取題目的法律元數據：考點、關鍵字和法條引用
    
    Args:
        question_content: 題目內容
        explanation: 題目解析（如果有）
    
    Returns:
        包含考點、關鍵字和法條引用的字典
    """
    try:
        # 從環境變量獲取OpenAI API密鑰
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("缺少OpenAI API密鑰")
            return {"exam_point": "", "keywords": [], "law_references": []}
        
        # 導入OpenAI庫
        try:
            from openai import OpenAI
        except ImportError:
            print("缺少OpenAI庫，嘗試安裝...")
            import subprocess
            subprocess.check_call(["pip", "install", "openai"])
            from openai import OpenAI
        
        # 創建OpenAI客戶端
        client = OpenAI(api_key=openai_api_key)
        
        # 構建提示詞
        prompt = f"""
        作為專業法律AI助手，請分析以下台灣法律考題及其解析，提取考點和關鍵字：
        
        題目：{question_content}
        解析：{explanation}
        
        請提取：
        1. 具體考點（需具體明確，如「無權代理之撤回權」，避免過於廣泛如「民法」）
        2. 關鍵字列表（包含核心法律概念、相關法條、情境或例外條件）
        3. 法條引用（如有，格式為「民法第171條」）
        
        以JSON格式返回：
        {{
          "exam_point": "具體考點",
          "keywords": ["關鍵字1", "關鍵字2", "關鍵字3"],
          "law_references": ["民法第171條"]
        }}
        
        注意：關鍵字需要精確且專業，避免過於寬泛的概念。每個關鍵字應該是具體的法律概念，而非籠統的分類。
        """
        
        # 發送請求到OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一個專業的法律助手，精通台灣法律。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        # 獲取回應文本
        response_text = response.choices[0].message.content
        
        # 清理回應文本，確保只包含JSON部分
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text.strip()
        
        try:
            result = json.loads(json_text)
            
            # 確保結果包含所有必要的字段
            if "exam_point" not in result:
                result["exam_point"] = ""
            if "keywords" not in result:
                result["keywords"] = []
            if "law_references" not in result:
                result["law_references"] = []
                
            return result
        except json.JSONDecodeError as e:
            print(f"無法解析JSON: {str(e)}, 回應文本: {response_text}")
            # 如果JSON解析失敗，嘗試使用正則表達式提取信息
            try:
                import re
                exam_point_match = re.search(r'"exam_point"\s*:\s*"([^"]+)"', response_text)
                exam_point = exam_point_match.group(1) if exam_point_match else ""
                
                keywords_match = re.search(r'"keywords"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
                keywords_str = keywords_match.group(1) if keywords_match else ""
                keywords = [k.strip().strip('"') for k in keywords_str.split(",") if k.strip()]
                
                law_refs_match = re.search(r'"law_references"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
                law_refs_str = law_refs_match.group(1) if law_refs_match else ""
                law_references = [l.strip().strip('"') for l in law_refs_str.split(",") if l.strip()]
                
                return {
                    "exam_point": exam_point,
                    "keywords": keywords,
                    "law_references": law_references
                }
            except Exception as regex_error:
                print(f"使用正則表達式提取數據時出錯: {str(regex_error)}")
                return {"exam_point": "", "keywords": [], "law_references": []}
            
    except Exception as e:
        print(f"使用OpenAI提取法律元數據時發生錯誤: {str(e)}")
        traceback.print_exc()
        # 發生錯誤時返回空結果
        return {"exam_point": "", "keywords": [], "law_references": []}

def extract_legal_metadata_hybrid(question_content, explanation=""):
    """
    混合策略提取法律元數據，動態切換API：
    1. 如果當前批次已決定使用OpenAI，則直接使用OpenAI API
    2. 否則先嘗試使用Gemini，如果失敗則切換到OpenAI
    3. 一旦切換到OpenAI並成功，就在當前批次中繼續使用OpenAI
    """
    global USE_OPENAI_FOR_CURRENT_BATCH
    
    # 如果當前批次已決定使用OpenAI，則直接使用OpenAI API
    if USE_OPENAI_FOR_CURRENT_BATCH:
        print("根據先前決策，直接使用OpenAI API...")
        return extract_legal_metadata_with_openai(question_content, explanation)
    
    # 先使用增強的降級機制
    fallback_result = enhanced_fallback_extraction(question_content, explanation)
    
    # 檢查降級結果是否足夠豐富
    if len(fallback_result["keywords"]) >= 2 and fallback_result["exam_point"]:
        print("使用降級機制提取到足夠的法律元數據")
        return fallback_result
    
    # 嘗試使用Gemini API
    try:
        print("嘗試使用Gemini API提取法律元數據...")
        gemini_result = extract_legal_metadata_with_retry(question_content, explanation)
        
        # 檢查Gemini結果是否有效
        if gemini_result.get("exam_point") or len(gemini_result.get("keywords", [])) > 0:
            print("使用Gemini API成功提取法律元數據")
            return gemini_result
        else:
            print("Gemini API返回的結果不夠豐富，嘗試使用OpenAI API")
            # 切換到OpenAI
            USE_OPENAI_FOR_CURRENT_BATCH = True
    except Exception as e:
        print(f"Gemini API提取失敗: {str(e)}，嘗試使用OpenAI API")
        # 切換到OpenAI
        USE_OPENAI_FOR_CURRENT_BATCH = True
    
    # 嘗試使用OpenAI API
    try:
        print("使用OpenAI API提取法律元數據...")
        openai_result = extract_legal_metadata_with_openai(question_content, explanation)
        if openai_result.get("exam_point") or len(openai_result.get("keywords", [])) > 0:
            print("OpenAI API成功提取法律元數據，後續將繼續使用OpenAI")
            # 確保後續使用OpenAI
            USE_OPENAI_FOR_CURRENT_BATCH = True
            return openai_result
    except Exception as e:
        print(f"OpenAI API提取失敗: {str(e)}，使用降級機制結果")
    
    # 如果所有API都失敗，返回降級結果
    return fallback_result

def process_questions_in_batches(questions_list):
    """
    處理批次處理後的題目列表:
    1. 確保題號連續性
    2. 去除重複題目
    3. 確保每個題目的字段完整
    4. 添加法律元數據
    """
    global USE_OPENAI_FOR_CURRENT_BATCH
    
    # 重置API選擇，每次處理從Gemini開始
    USE_OPENAI_FOR_CURRENT_BATCH = False
    
    if not questions_list:
        return []
    
    # 去除重複題目和整理題號
    unique_questions = {}
    for question in questions_list:
        # 跳過無效數據
        if not isinstance(question, dict) or "content" not in question:
            continue
            
        # 取得題號
        q_id = question.get("id")
        if not q_id or not isinstance(q_id, int):
            continue
            
        # 使用題號作為唯一標識，保留最完整的版本
        if q_id not in unique_questions or len(question.get("content", "")) > len(unique_questions[q_id].get("content", "")):
            # 確保所有必要字段都存在且不為None
            if "type" not in question or question["type"] is None:
                question["type"] = "單選題"  # 默認為單選題
            if "options" not in question or question["options"] is None:
                question["options"] = {}
            if "answer" not in question or question["answer"] is None:
                question["answer"] = ""
            if "explanation" not in question or question["explanation"] is None:
                question["explanation"] = ""
            if "exam_point" not in question or question["exam_point"] is None:
                question["exam_point"] = ""
            if "keywords" not in question or question["keywords"] is None:
                question["keywords"] = []
            if "law_references" not in question or question["law_references"] is None:
                question["law_references"] = []
                
            unique_questions[q_id] = question
    
    # 按題號排序
    sorted_questions = sorted(unique_questions.values(), key=lambda q: q.get("id", 0))
    
    # 重新分配題號確保連續性
    for i, question in enumerate(sorted_questions):
        question["id"] = i + 1
    
    # 分批處理以提取法律元數據
    processed_questions = []
    batch_size = 5
    
    for i in range(0, len(sorted_questions), batch_size):
        batch = sorted_questions[i:i+batch_size]
        print(f"處理第 {i//batch_size + 1}/{(len(sorted_questions)-1)//batch_size + 1} 批問題的法律元數據...")
        
        for question in batch:
            # 提取法律元數據（如果尚未有）
            if (not question.get("exam_point") and not question.get("keywords") and 
                not question.get("law_references")):
                explanation = question.get("explanation", "")
                try:
                    legal_metadata = extract_legal_metadata_hybrid(question["content"], explanation)
                    
                    # 添加法律元數據到題目，確保沒有None值
                    question["exam_point"] = legal_metadata.get("exam_point", "") or ""
                    question["keywords"] = legal_metadata.get("keywords", []) or []
                    question["law_references"] = legal_metadata.get("law_references", []) or []
                except Exception as e:
                    print(f"提取法律元數據時發生錯誤: {str(e)}")
                    # 確保字段存在
                    question["exam_point"] = question.get("exam_point", "") or ""
                    question["keywords"] = question.get("keywords", []) or []
                    question["law_references"] = question.get("law_references", []) or []
            
            processed_questions.append(question)
        
        # 批次處理完成，稍等一段時間避免API調用過度集中
        if i + batch_size < len(sorted_questions):
            import random
            import time
            wait_time = random.uniform(1.0, 2.0)
            print(f"批次處理完成，等待 {wait_time:.1f} 秒...")
            time.sleep(wait_time)
    
    # 修改這部分，使用計數器而不是每次都輸出
    success_count = 0
    fallback_count = 0
    
    for question in processed_questions:
        # 提取法律元數據
        try:
            legal_metadata = extract_legal_metadata_hybrid(question["content"], explanation)
            # 成功計數+1而不是每次都輸出
            success_count += 1
        except Exception as e:
            fallback_count += 1
            # ...處理異常...
    
    # 批處理結束後只輸出一次匯總信息
    print(f"批次處理完成: 成功 {success_count} 題，降級處理 {fallback_count} 題")
    
    return processed_questions

def split_text_into_blocks(text, block_size=4000):
    """
    將長文本分割成多個較小的區塊，以適應API處理限制
    
    Args:
        text: 要分割的文本
        block_size: 每個區塊的最大字元數
        
    Returns:
        分割後的文本區塊列表
    """
    if not text:
        return []
    
    # 使用段落或句號作為分隔點，避免切割到句子中間
    paragraphs = text.split('\n\n')
    blocks = []
    current_block = ""
    
    for para in paragraphs:
        # 如果段落自身超過塊大小，進一步按句號分割
        if len(para) > block_size:
            sentences = para.replace('。', '。\n').split('\n')
            for sentence in sentences:
                if len(current_block) + len(sentence) <= block_size:
                    current_block += sentence
                else:
                    if current_block:
                        blocks.append(current_block)
                    current_block = sentence
        else:
            # 檢查加入此段落是否會超出塊大小
            if len(current_block) + len(para) + 2 <= block_size:  # +2 為段落間的換行符
                if current_block:
                    current_block += "\n\n" + para
                else:
                    current_block = para
            else:
                if current_block:
                    blocks.append(current_block)
                current_block = para
    
    # 添加最後一個塊（如果有）
    if current_block:
        blocks.append(current_block)
    
    return blocks

def process_block(block_text, block_num, start_id=1):
    """
    處理單個文本區塊，提取法律題目
    
    Args:
        block_text: 文本區塊內容
        block_num: 區塊編號（用於日誌）
        start_id: 題目ID起始值
        
    Returns:
        提取的題目列表
    """
    try:
        # 建立Gemini模型實例
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # 構造提示詞
        prompt = f"""
        你是一位專精於台灣法律考試的人工智能，善於從考試資料中精確識別和結構化法律題目。
        請從下面的文本中提取所有法律考試題目，並以JSON格式返回。從題號 {start_id} 開始編號。

        【任務說明】
        1. 仔細分析文本，識別所有獨立的考試題目
        2. 對每個題目進行分類（單選題、多選題、簡答題、申論題等）
        3. 提取題目內容、選項、答案和解析（如有）
        4. 返回結構化的JSON數據

        【格式要求】
        ```json
        {{
          "questions": [
            {{
              "id": {start_id},
              "type": "單選題", // 可能的值: 單選題、多選題、簡答題、申論題、是非題
              "content": "完整題幹內容，包含題號和題目描述",
              "options": {{"A": "選項A內容", "B": "選項B內容", "C": "選項C內容", "D": "選項D內容"}},
              "answer": "A", // 單選題填寫選項字母，多選題用逗號分隔，如"A,C"
              "explanation": "解析內容（如文本中提供）"
            }}
            // ... 更多題目
          ]
        }}
        ```

        【特別注意】
        1. 題型判斷:
           - 單選題通常指示「下列何者」或明確表明只有一個答案
           - 多選題通常表述為「下列何者為正確」或明確表明有多個答案
           - 簡答題通常要求簡短文字回答
           - 申論題通常需要長篇分析討論
           - 是非題通常只需判斷對錯

        2. 答案處理:
           - 單選題：只填寫一個選項字母（如"A"）
           - 多選題：用逗號分隔多個選項字母（如"A,C,D"）
           - 其他題型：填入文字答案或保留空字串

        3. 選項格式:
           - 確保選項使用正確的鍵值對格式
           - 如果是簡答題或申論題等沒有選項的題型，options應為空對象 {{}}

        請分析以下考試文本:
        {block_text}
        
        僅返回有效的JSON格式，不要加入其他說明文字。如果無法識別任何完整題目，返回空的questions陣列。
        """
        
        # 發送請求到API
        response = model.generate_content(prompt)
        response_text = response.text
        
        # 清理回應文本，確保只包含JSON部分
        import json
        import re
        
        # 移除可能的markdown代碼塊標記
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text.strip()
        
        # 嘗試解析JSON
        try:
            result = json.loads(json_text)
            
            # 檢查結果格式
            if "questions" in result and isinstance(result["questions"], list):
                print(f"從區塊 {block_num} 成功提取 {len(result['questions'])} 道題目")
                
                # 對每個題目進行基本驗證和清理
                valid_questions = []
                for q in result["questions"]:
                    # 跳過無效題目
                    if not isinstance(q, dict) or "content" not in q or not q["content"]:
                        continue
                    
                    # 確保每個題目都有必要的字段
                    if "type" not in q:
                        q["type"] = "單選題"  # 默認為單選題
                    if "options" not in q or q["options"] is None:
                        q["options"] = {}
                    if "answer" not in q:
                        q["answer"] = ""
                    if "explanation" not in q:
                        q["explanation"] = ""
                    
                    valid_questions.append(q)
                
                return valid_questions
            else:
                print(f"區塊 {block_num} 返回的JSON缺少questions字段或格式不正確")
                return []
                
        except json.JSONDecodeError as e:
            print(f"無法解析區塊 {block_num} 的JSON: {str(e)}")
            
            # 嘗試使用正則表達式提取JSON
            try:
                pattern = r'\{\s*"questions"\s*:\s*\[(.*?)\]\s*\}'
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    fixed_json = '{"questions": [' + match.group(1) + ']}'
                    result = json.loads(fixed_json)
                    if "questions" in result and isinstance(result["questions"], list):
                        print(f"通過正則修復後，成功從區塊 {block_num} 提取 {len(result['questions'])} 道題目")
                        return result["questions"]
            except Exception:
                pass
            
            # 如果上述方法都失敗，返回空列表
            return []
            
    except Exception as e:
        print(f"處理區塊 {block_num} 時發生錯誤: {str(e)}")
        return []

def process_text_with_gemini(extracted_text, filename):
    """
    使用Gemini API處理PDF提取的文字
    
    Args:
        extracted_text: 從PDF提取的文字內容
        filename: PDF檔案名（用於日誌）
        
    Returns:
        處理後的題目列表或包含警告的字典
    """
    # 導入模組
    import time
    
    if not extracted_text:
        return {"warning": "沒有提供有效的文字內容進行處理"}
    
    # 將大文本分割成多個區塊，每個區塊約4000字符
    blocks = split_text_into_blocks(extracted_text, 4000)
    if not blocks:
        return {"warning": "文字分割後沒有有效內容"}
    
    print(f"將文字分割為 {len(blocks)} 個區塊進行處理")
    
    # 批次處理每個區塊
    all_questions = []
    failed_blocks = []
    current_id = 1
    
    for i, block in enumerate(blocks):
        if block is None or len(block.strip()) < 50:
            print(f"跳過區塊 {i+1}，內容為空或太短")
            continue
            
        try:
            print(f"處理區塊 {i+1}/{len(blocks)}")
            
            # 嘗試處理區塊，最多重試3次
            max_retries = 3
            retry_delay = 2  # 初始延遲2秒
            
            for attempt in range(max_retries):
                try:
                    block_questions = process_block(block, i+1, current_id)
                    
                    # 成功處理，更新當前ID並添加題目
                    if block_questions:
                        all_questions.extend(block_questions)
                        # 更新當前ID為最大ID+1
                        max_id = max([q.get("id", 0) for q in block_questions]) if block_questions else 0
                        current_id = max_id + 1
                    
                    # 成功處理，退出重試循環
                    break
                    
                except ResourceExhausted as quota_error:
                    # 如果是配額錯誤且已經是最後一次嘗試
                    if attempt == max_retries - 1:
                        print(f"區塊 {i+1} 處理失敗: 配額限制 ({quota_error})")
                        failed_blocks.append(i)
                        # 配額用完但已有結果，返回部分結果
                        if all_questions:
                            return {
                                "questions": all_questions,
                                "warning": f"API配額已用完，只處理了 {i}/{len(blocks)} 個區塊"
                            }
                        # 配額用完且無結果，拋出異常
                        raise Exception(f"API配額已用完") from quota_error
                    
                    # 未達到最大重試次數，等待後重試
                    print(f"區塊 {i+1} 遇到配額限制，等待 {retry_delay} 秒後重試 ({attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指數退避
                
                except Exception as e:
                    # 如果是其他錯誤且已經是最後一次嘗試
                    if attempt == max_retries - 1:
                        print(f"區塊 {i+1} 處理出錯: {str(e)}")
                        failed_blocks.append(i)
                        # 繼續處理其他區塊
                        break
                    
                    # 未達到最大重試次數，等待後重試
                    print(f"區塊 {i+1} 處理出錯，等待 {retry_delay} 秒後重試 ({attempt+1}/{max_retries}): {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指數退避
        
        except Exception as block_error:
            print(f"處理區塊 {i+1} 時出錯: {str(block_error)}")
            failed_blocks.append(i)
            
            # 如果是配額錯誤且已有部分結果
            if "配額" in str(block_error) or "quota" in str(block_error).lower() or "429" in str(block_error):
                if all_questions:
                    return {
                        "questions": all_questions,
                        "warning": f"API配額已用完，只處理了 {i}/{len(blocks)} 個區塊"
                    }
    
    # 處理完成後的總結
    if failed_blocks:
        failed_blocks_str = ", ".join(map(str, failed_blocks))
        if all_questions:
            return {
                "questions": all_questions,
                "warning": f"部分區塊處理失敗（區塊: {failed_blocks_str}），僅返回成功處理的 {len(all_questions)} 個題目"
            }
        else:
            return {"warning": f"所有區塊處理失敗（區塊: {failed_blocks_str}），無法獲取題目"}
    
    # 對題號進行整理，確保連續性
    all_questions = process_questions_in_batches(all_questions)
    
    return all_questions

def evaluate_exam_point_coverage(generated_answers, question_exam_points):
    coverage_metrics = {
        "total_coverage": 0,
        "primary_coverage": 0,
        "complete_coverage_ratio": 0
    }
    
    total_questions = len(generated_answers)
    complete_coverage_count = 0
    
    for i, answer in enumerate(generated_answers):
        # 提取答案中的考點
        extracted_points = extract_exam_points(answer)
        # 標準考點（區分主要和次要考點）
        standard_points = question_exam_points[i]
        primary_points = [p for p in standard_points if p["importance"] == "primary"]
        
        # 計算覆蓋率
        covered_points = set(extracted_points).intersection(set([p["name"] for p in standard_points]))
        covered_primary = set(extracted_points).intersection(set([p["name"] for p in primary_points]))
        
        total_coverage = len(covered_points) / len(standard_points) if standard_points else 1
        primary_coverage = len(covered_primary) / len(primary_points) if primary_points else 1
        
        # 計入完全覆蓋的數量
        if total_coverage > 0.9 and primary_coverage == 1:
            complete_coverage_count += 1
            
        coverage_metrics["total_coverage"] += total_coverage
        coverage_metrics["primary_coverage"] += primary_coverage
    
    # 計算平均值
    coverage_metrics["total_coverage"] /= total_questions
    coverage_metrics["primary_coverage"] /= total_questions
    coverage_metrics["complete_coverage_ratio"] = complete_coverage_count / total_questions
    
    return coverage_metrics

def extract_exam_points(answer_text):
    """
    從回答文本中提取考點
    
    Args:
        answer_text: 回答文本內容
        
    Returns:
        提取出的考點列表
    """
    # 方法一：使用簡單關鍵詞匹配
    exam_points = []
    
    # 如果答案中明確標記了考點部分
    if "考點：" in answer_text or "考點:" in answer_text:
        parts = re.split(r'考點[：:]', answer_text)
        if len(parts) > 1:
            point_section = parts[1].strip()
            # 提取到下一個標題或段落結束
            end_markers = ["關鍵詞", "法條", "解析", "說明"]
            for marker in end_markers:
                if marker in point_section:
                    point_section = point_section.split(marker)[0]
            
            # 分割成多個考點（如有多個）
            points = [p.strip() for p in re.split(r'[,，、；;]', point_section) if p.strip()]
            exam_points.extend(points)
    
    # 如果沒有找到明確標記的考點，嘗試從整個文本中提取
    if not exam_points:
        # 這裡可以實現更複雜的提取邏輯，例如使用LLM或關鍵詞匹配
        pass
    
    return exam_points

if __name__ == "__main__":
    # 測試用範例
    with open("sample.pdf", "rb") as f:
        pdf_bytes = f.read()
    result = process_pdf_with_gemini(pdf_bytes, "sample.pdf")
    print(json.dumps(result, ensure_ascii=False, indent=2))