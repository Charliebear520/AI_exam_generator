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
import numpy as np
import sys

# 嘗試導入OpenAI包
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI庫未安裝，將不能使用OpenAI功能")

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

# 模型配置和批次處理參數
GEMINI_MODELS = {
    "text": {
        "default": "gemini-2.0-flash-lite",  # 使用Flash-Lite模型（更高RPM限制4000）
        "fallback": "gemini-1.5-flash",  # 備用模型
        "premium": "gemini-2.5-pro-exp-03-25",  # 有條件時使用的高級模型（低RPM限制）
        "batch_size": 25,  # 每批處理的題目數量，針對Flash模型調整
        "rpm_limit": 4000  # 更新為Flash-Lite的RPM限制
    },
    "embedding": {
        "default": "gemini-embedding-exp",
        "fallback": "all-MiniLM-L6-v2",  # 降級到SentenceTransformer
        "batch_size": 8,  # 嵌入模型批次大小
        "rpm_limit": 10
    }
}

# 退避策略參數
BACKOFF_CONFIG = {
    "gemini-2.0-flash": {
        "initial_wait": 0.5,  # 初始等待時間(秒)
        "max_retries": 5,     # 最大重試次數
        "backoff_factor": 1.5 # 退避倍數
    },
    "gemini-2.5-pro-exp-03-25": {
        "initial_wait": 5.0,  # 初始等待更長
        "max_retries": 2,     # 重試次數少，避免迅速耗盡每日配額
        "backoff_factor": 2.0 # 退避倍數更大
    },
    "gemini-embedding-exp": {
        "initial_wait": 1.0,
        "max_retries": 3,
        "backoff_factor": 2.0
    },
    "default": {
        "initial_wait": 1.0,
        "max_retries": 3,
        "backoff_factor": 2.0
    }
}

# 智能降級閾值設置
METADATA_QUALITY_THRESHOLD = {
    "keywords_min": 2,  # 至少需要2個關鍵詞
    "exam_point_min_length": 5,  # 考點最小長度
    "law_refs_preferred": True  # 優先考慮有法條引用的結果
}

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

def extract_legal_metadata(question_content, explanation="", model_name=None, multi_doc=False):
    """
    提取文本的法律元數據，支持指定模型和多文檔處理
    
    Args:
        question_content: 要分析的文本內容
        explanation: 補充說明或解析 (針對題目時使用)
        model_name: 指定使用的模型名稱
        multi_doc: 是否作為多文檔處理
        
    Returns:
        包含元數據的字典
    """
    try:
        # 如果未指定模型，使用默認模型
        if not model_name:
            model_name = GEMINI_MODELS["text"]["default"]
            
        # 處理多文檔模式
        if multi_doc:
            # 使用文檔結構提取函數
            result = extract_document_structure(question_content, model_name)
        else:
            # 檢查是否是題目與解析格式或一般文本
            if explanation:
                # 使用原有題目元數據提取邏輯
                model = genai.GenerativeModel(model_name)
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
                - 考點必須具體明確，不要過於籠統
                - 關鍵字必須是法律術語，避免使用一般性詞彙
                - 如無明確法條，可以使用相關法律原則或學說
                """.format(question=question_content, explanation=explanation)
                
                response = model.generate_content(prompt)
                result_text = response.text
                
                # 提取JSON字串
                import re
                import json
                
                json_match = re.search(r'\{[\s\S]*\}', result_text)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError:
                        # 嘗試修復JSON
                        json_str = json_str.replace("'", '"')
                        try:
                            result = json.loads(json_str)
                        except:
                            raise ValueError(f"無法解析Gemini返回的JSON: {json_str}")
                else:
                    # 如果找不到JSON，嘗試更靈活的提取方式
                    exam_point_match = re.search(r'"exam_point"\s*:\s*"([^"]+)"', result_text)
                    keywords_match = re.search(r'"keywords"\s*:\s*\[(.*?)\]', result_text, re.DOTALL)
                    law_refs_match = re.search(r'"law_references"\s*:\s*\[(.*?)\]', result_text, re.DOTALL)
                    
                    result = {}
                    
                    if exam_point_match:
                        result["exam_point"] = exam_point_match.group(1)
                    else:
                        result["exam_point"] = ""
                        
                    if keywords_match:
                        keywords_str = keywords_match.group(1)
                        keywords = re.findall(r'"([^"]+)"', keywords_str)
                        result["keywords"] = keywords
                    else:
                        result["keywords"] = []
                        
                    if law_refs_match:
                        law_refs_str = law_refs_match.group(1)
                        law_refs = re.findall(r'"([^"]+)"', law_refs_str)
                        result["law_references"] = law_refs
                    else:
                        result["law_references"] = []
            else:
                # 一般文本，使用單文檔元數據提取函數
                result = extract_metadata_single_doc(question_content, model_name)
        
        # 確保結果是可JSON序列化的
        result = clean_metadata_for_json(result)
        return result
        
    except Exception as e:
        print(f"使用 {model_name} 提取法律元數據失敗: {str(e)}")
        # 返回錯誤信息而不是拋出異常，確保處理流程不中斷
        return {
            "error": str(e),
            "status": "failed", 
            "text_preview": question_content[:100] + "..." if len(question_content) > 100 else question_content
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
            
            # 清理數據中的None值和其他不可序列化的值
            if result:
                try:
                    # 在生成JSON前進行全面清理
                    result = clean_metadata_for_json(result)
                    
                    # 對每個問題進行額外清理和檢查
                    if "questions" in result and isinstance(result["questions"], list):
                        for q in result["questions"]:
                            # 確保每個問題有所有必需的字段
                            if "id" not in q:
                                q["id"] = random.randint(1000, 9999)  # 生成隨機ID
                            if "content" not in q or not q["content"]:
                                q["content"] = "未能識別題目內容"
                            if "options" not in q or not isinstance(q["options"], dict):
                                q["options"] = {}
                            if "answer" not in q:
                                q["answer"] = ""
                
                    # 額外檢查是否有語法錯誤
                    try:
                        # 測試序列化
                        json_str = json.dumps(result, ensure_ascii=False)
                    except Exception as json_error:
                        print(f"序列化錯誤: {str(json_error)}")
                        # 如果序列化失敗，嘗試更嚴格的清理
                        result = {"questions": [], "warning": f"序列化失敗: {str(json_error)}"}
                except Exception as clean_error:
                    print(f"清理數據時發生錯誤: {str(clean_error)}")
                    # 提供一個安全的回退結果
                    result = {"questions": [], "warning": f"數據清理錯誤: {str(clean_error)}"}
            
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
    """
    本地提取法律元數據的後備方案，不依賴外部API
    在所有API請求嘗試均失敗時使用
    """
    import re
    from collections import Counter
    
    # 初始化空結果
    metadata = {
        "exam_point": "",
        "keywords": [],
        "law_references": []
    }
    
    # 合併內容和解析用於提取
    combined_text = question_content + " " + explanation
    
    # 1. 提取法條引用
    # 匹配常見法律法條格式
    law_patterns = [
        r'([^，。、；\s]*法)[第條][一二三四五六七八九十百千0-9０-９]+(條|項|款|目)',
        r'([^，。、；\s]*法)第([0-9０-９]+)條',
        r'([^，。、；\s]*法)(第[0-9０-９]+條)',
        r'《([^》]+)》第([0-9０-９]+)條'
    ]
    
    law_refs = []
    for pattern in law_patterns:
        matches = re.findall(pattern, combined_text)
        for match in matches:
            if isinstance(match, tuple):
                law_name = match[0]
                if len(match) > 1 and match[1]:
                    try:
                        article = match[1]
                        law_refs.append(f"{law_name}第{article}")
                    except:
                        law_refs.append(law_name)
                else:
                    law_refs.append(law_name)
            else:
                law_refs.append(match)
    
    # 去重
    metadata["law_references"] = list(set(law_refs))
    
    # 2. 提取關鍵詞
    # 常見法律詞彙
    legal_terms = [
        "代理", "侵權", "物權", "債權", "損害賠償", "契約", "買賣", "贈與", "租賃", 
        "借貸", "請求權", "所有權", "抵押權", "質權", "用益物權", "占有", "時效", 
        "正當防衛", "緊急避難", "不當得利", "無因管理", "詐欺", "脅迫", "善意", 
        "惡意", "撤銷", "撤回", "無效", "效力未定", "消滅時效", "除斥期間", "連帶",
        "債務不履行", "瑕疵擔保", "人格權", "國家賠償", "消費者保護", "保證", "民事",
        "刑事", "行政", "訴訟", "上訴", "權利", "義務", "責任", "過失", "故意"
    ]
    
    # 從解析中提取關鍵詞
    keywords = []
    for term in legal_terms:
        if term in combined_text:
            keywords.append(term)
    
    # 提取特殊關鍵詞
    special_patterns = [
        r'「([^」]{2,10})」',  # 引號中的2-10個字符
        r'『([^』]{2,10})』',  # 雙引號中的2-10個字符
        r'（([^）]{2,15})）'   # 括號中的2-15個字符
    ]
    
    for pattern in special_patterns:
        matches = re.findall(pattern, combined_text)
        for match in matches:
            if len(match) >= 2 and any(term in match for term in legal_terms):
                keywords.append(match)
    
    # 最多保留5個關鍵詞
    metadata["keywords"] = list(set(keywords))[:5]
    
    # 3. 提取考點
    # 根據關鍵詞和法條引用推斷考點
    if metadata["keywords"] and metadata["law_references"]:
        main_law = metadata["law_references"][0] if metadata["law_references"] else ""
        main_keyword = metadata["keywords"][0] if metadata["keywords"] else ""
        
        # 嘗試從解析中找出關鍵句
        sentences = re.split(r'[。！？]', explanation)
        important_sentences = []
        for sentence in sentences:
            if main_keyword in sentence or (main_law and main_law in sentence):
                important_sentences.append(sentence)
        
        if important_sentences:
            # 找出最短但有意義的句子作為考點
            valid_sentences = [s for s in important_sentences if len(s) > 5]
            if valid_sentences:
                best_sentence = min(valid_sentences, key=len)
                metadata["exam_point"] = best_sentence.strip()[:30]  # 限制長度
            else:
                # 關鍵詞 + 法條
                metadata["exam_point"] = f"{main_keyword}與{main_law}適用"
        else:
            # 沒有找到重要句子，組合關鍵詞和法條
            metadata["exam_point"] = f"{main_keyword}相關法律適用"
    elif metadata["keywords"]:
        # 只有關鍵詞
        metadata["exam_point"] = f"{metadata['keywords'][0]}的法律適用"
    elif metadata["law_references"]:
        # 只有法條
        metadata["exam_point"] = f"{metadata['law_references'][0]}的適用"
    else:
        # 完全沒有提取到有用信息
        # 嘗試根據問題類型判斷
        if "契約" in combined_text or "合約" in combined_text:
            metadata["exam_point"] = "契約法律關係"
        elif "侵權" in combined_text or "損害" in combined_text:
            metadata["exam_point"] = "侵權行為法律關係"
        elif "刑法" in combined_text or "犯罪" in combined_text:
            metadata["exam_point"] = "刑法基本概念"
        else:
            metadata["exam_point"] = "法律適用原則"
    
    print(f"使用本地後備方法提取元數據: {metadata}")
    return metadata

def extract_legal_metadata_with_openai(question_content, explanation="", model_name="gpt-3.5-turbo"):
    """
    使用OpenAI API提取法律元數據作為降級處理
    
    Args:
        question_content: 文本內容
        explanation: 解析或說明
        model_name: OpenAI模型名稱
        
    Returns:
        包含法律元數據的字典
    """
    try:
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
        {{
          "exam_point": "具體法律考點",
          "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3", "關鍵詞4", "關鍵詞5"],
          "law_references": ["民法第X條", "民法第Y條"]
        }}
        """.format(question=question_content, explanation=explanation)
        
        print(f"使用OpenAI API ({model_name}) 提取法律元數據...")
        
        # 使用通用的OpenAI處理函數
        result_text = process_with_openai(
            prompt=prompt,
            model_name=model_name,
            temperature=0.3,
            max_tokens=800
        )
        
        # 提取JSON
        metadata = extract_json_from_text(result_text)
        
        # 如果提取失敗，使用備選提取方式
        if not metadata:
            print("無法從OpenAI響應中提取JSON，嘗試使用備選提取方式")
            return enhanced_fallback_extraction(question_content, explanation)
            
        # 確保處理後的元數據可以被JSON序列化
        metadata = clean_metadata_for_json(metadata)
        return metadata
            
    except Exception as e:
        print(f"使用OpenAI API提取元數據時出錯: {str(e)}")
        # 使用本地提取作為最後的備用選項
        return enhanced_fallback_extraction(question_content, explanation)

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
    優化的批次處理函數 - 根據免費配額限制調整
    
    根據不同API類型自動調整批次大小和策略
    """
    # 使用配置中定義的批次大小
    batch_size = GEMINI_MODELS["text"]["batch_size"]
    
    # 根據問題總數動態調整批次大小
    total_questions = len(questions_list)
    if total_questions < 30:
        # 小批量處理，減小批次大小
        batch_size = min(10, batch_size)
    elif total_questions > 100:
        # 大批量處理，適度增加批次大小但不超過配置值
        batch_size = min(30, batch_size)
    
    print(f"處理 {total_questions} 個問題，批次大小設為 {batch_size}")
    
    # 將題目分成多個批次
    batches = [questions_list[i:i+batch_size] for i in range(0, len(questions_list), batch_size)]
    
    # 追蹤當前使用哪個API和成功率
    api_performance = {
        "gemini": {"success": 0, "failure": 0},
        "openai": {"success": 0, "failure": 0},
        "local": {"success": 0, "failure": 0}
    }
    current_api = "gemini"  # 初始使用Gemini
    
    # 處理每個批次
    results = []
    for batch_idx, batch in enumerate(batches, 1):
        print(f"處理第 {batch_idx}/{len(batches)} 批問題的法律元數據...")
        
        # 分析前面批次的成功率，決定使用哪個API
        if batch_idx > 1:
            # 計算Gemini的成功率
            gemini_success_rate = 0
            if api_performance["gemini"]["success"] + api_performance["gemini"]["failure"] > 0:
                gemini_success_rate = api_performance["gemini"]["success"] / (
                    api_performance["gemini"]["success"] + api_performance["gemini"]["failure"]
                )
            
            # 計算OpenAI的成功率
            openai_success_rate = 0
            if api_performance["openai"]["success"] + api_performance["openai"]["failure"] > 0:
                openai_success_rate = api_performance["openai"]["success"] / (
                    api_performance["openai"]["success"] + api_performance["openai"]["failure"]
                )
            
            # 根據成功率選擇API
            if current_api == "gemini" and gemini_success_rate < 0.6 and openai_success_rate > gemini_success_rate:
                current_api = "openai"
                print(f"Gemini成功率低({gemini_success_rate:.2f})，切換到OpenAI API")
            elif current_api == "openai" and openai_success_rate < 0.4 and gemini_success_rate > 0.7:
                current_api = "gemini"
                print(f"OpenAI成功率低，恢復使用Gemini API")
        
        # 根據選定的API處理當前批次
        batch_results = []
        batch_success = 0
        batch_failure = 0
        
        for question in batch:
            content = question.get("content", "")
            explanation = question.get("explanation", "")
            
            try:
                # 嘗試提取法律元數據
                if current_api == "gemini":
                    print("嘗試使用Gemini API提取法律元數據...")
                    metadata = extract_legal_metadata_with_retry(
                        content, explanation, 
                        max_retries=BACKOFF_CONFIG["default"]["max_retries"],
                        backoff_factor=BACKOFF_CONFIG["default"]["backoff_factor"],
                        initial_wait=BACKOFF_CONFIG["default"]["initial_wait"]
                    )
                    
                    # 評估元數據質量
                    quality_score = evaluate_metadata_quality(metadata)
                    if quality_score < 0.7:  # 質量分數閾值
                        print(f"Gemini結果質量不佳(分數{quality_score:.2f})，嘗試使用OpenAI")
                        metadata = extract_legal_metadata_with_openai(content, explanation)
                        api_performance["gemini"]["failure"] += 1
                        api_performance["openai"]["success"] += 1
                        batch_failure += 1
                    else:
                        api_performance["gemini"]["success"] += 1
                        batch_success += 1
                        
                elif current_api == "openai":
                    print("根據先前決策，直接使用OpenAI API...")
                    metadata = extract_legal_metadata_with_openai(content, explanation)
                    
                    # 評估元數據質量
                    quality_score = evaluate_metadata_quality(metadata)
                    if quality_score < 0.6:  # OpenAI結果質量閾值
                        print(f"OpenAI結果質量不佳(分數{quality_score:.2f})，嘗試本地提取")
                        fallback_metadata = enhanced_fallback_extraction(content, explanation)
                        # 合併結果，保留OpenAI的部分有效數據
                        metadata = merge_metadata_results(metadata, fallback_metadata)
                        api_performance["openai"]["failure"] += 1
                        api_performance["local"]["success"] += 1
                        batch_failure += 1
                    else:
                        api_performance["openai"]["success"] += 1
                        batch_success += 1
                        
                # 使用提取的元數據更新問題
                question.update(metadata)
                batch_results.append(question)
                
            except Exception as e:
                print(f"提取元數據時出錯: {str(e)}")
                # 使用本地提取作為最後的備用選項
                fallback_metadata = enhanced_fallback_extraction(content, explanation)
                question.update(fallback_metadata)
                batch_results.append(question)
                
                if current_api == "gemini":
                    api_performance["gemini"]["failure"] += 1
                elif current_api == "openai":
                    api_performance["openai"]["failure"] += 1
                batch_failure += 1
        
        # 輸出批次處理結果
        print(f"批次處理完成: 成功={batch_success}, 失敗={batch_failure}")
        
        # 添加批次結果到總結果
        results.extend(batch_results)
        
        # 批次間添加一個小延遲，避免觸發API限制
        if batch_idx < len(batches):
            import time
            wait_time = 1.3  # 批次間等待時間
            print(f"批次處理完成，等待 {wait_time} 秒...")
            time.sleep(wait_time)
    
    # 總結處理結果
    print(f"===== 元數據提取完成 =====")
    print(f"Gemini API: 成功={api_performance['gemini']['success']}, 失敗={api_performance['gemini']['failure']}")
    print(f"OpenAI API: 成功={api_performance['openai']['success']}, 失敗={api_performance['openai']['failure']}")
    print(f"本地提取: 使用次數={api_performance['local']['success']}")
    
    return results

# 添加元數據質量評估函數
def evaluate_metadata_quality(metadata):
    """
    評估提取的元數據質量，返回0-1之間的分數
    """
    score = 0.0
    
    # 檢查考點
    exam_point = metadata.get("exam_point", "")
    if exam_point and len(exam_point) >= METADATA_QUALITY_THRESHOLD["exam_point_min_length"]:
        score += 0.4
        # 額外加分：考點包含標準法律術語
        legal_terms = ["侵權", "契約", "損害賠償", "刑法", "民法", "權利", "義務", "代理", "所有權"]
        if any(term in exam_point for term in legal_terms):
            score += 0.1
    
    # 檢查關鍵詞
    keywords = metadata.get("keywords", [])
    if isinstance(keywords, list) and len(keywords) >= METADATA_QUALITY_THRESHOLD["keywords_min"]:
        keyword_score = min(len(keywords) / 5.0, 1.0) * 0.3  # 最多5個關鍵詞得滿分
        score += keyword_score
    
    # 檢查法條引用
    law_references = metadata.get("law_references", [])
    if isinstance(law_references, list) and len(law_references) > 0:
        score += 0.2
    
    return min(score, 1.0)  # 確保分數不超過1

# 合併多個元數據結果
def merge_metadata_results(primary, secondary):
    """
    合併兩個元數據結果，保留質量較好的部分
    """
    result = primary.copy()
    
    # 如果主要結果缺少考點，使用次要結果的考點
    if not result.get("exam_point") and secondary.get("exam_point"):
        result["exam_point"] = secondary["exam_point"]
    
    # 合併關鍵詞，去重
    primary_keywords = result.get("keywords", [])
    secondary_keywords = secondary.get("keywords", [])
    if isinstance(primary_keywords, list) and isinstance(secondary_keywords, list):
        combined_keywords = list(set(primary_keywords + secondary_keywords))
        result["keywords"] = combined_keywords
    
    # 合併法條引用，去重
    primary_laws = result.get("law_references", [])
    secondary_laws = secondary.get("law_references", [])
    if isinstance(primary_laws, list) and isinstance(secondary_laws, list):
        combined_laws = list(set(primary_laws + secondary_laws))
        result["law_references"] = combined_laws
    
    return result

# 修改元數據提取的重試函數
def extract_legal_metadata_with_retry(
    question_content, 
    explanation="", 
    max_retries=3, 
    backoff_factor=2.0,
    initial_wait=1.0
):
    """
    優化的元數據提取函數 - 根據配額限制調整退避策略
    """
    import time
    
    # 獲取當前使用的模型，預設使用Flash模型(高RPM)
    current_model = GEMINI_MODELS["text"]["default"]
    
    # 嘗試提取
    for retry in range(max_retries):
        try:
            # 如果是第一次嘗試，使用默認模型；否則嘗試備用模型
            if retry == 0:
                model_name = current_model
            else:
                model_name = GEMINI_MODELS["text"]["fallback"]
                
            # 標記當前嘗試的模型
            print(f"嘗試使用 {model_name} 提取法律元數據...")
            
            # 調用提取函數
            result = extract_legal_metadata(question_content, explanation, model_name)
            print(f"使用 {model_name} 成功提取法律元數據")
            return result
            
        except Exception as e:
            error_message = str(e).lower()
            
            # 檢查是否是配額限制相關錯誤
            if "quota" in error_message or "rate limit" in error_message or "too many requests" in error_message:
                # 計算等待時間（使用指數退避策略）
                wait_time = initial_wait * (backoff_factor ** retry)
                
                # 對於不同的模型使用不同的退避策略
                if "2.5" in current_model or "pro" in current_model.lower():
                    # 高級模型有較低的RPM，使用更長的等待時間
                    wait_time = wait_time * 2
                
                print(f"API配額限制，等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
                
                # 如果是最後一次重試且仍然失敗，切換到OpenAI
                if retry == max_retries - 1:
                    print("Gemini API返回的結果不夠豐富，嘗試使用OpenAI API")
                    return extract_legal_metadata_with_openai(question_content, explanation)
            else:
                # 非配額限制錯誤，直接嘗試OpenAI
                print(f"Gemini API發生錯誤: {error_message}")
                return extract_legal_metadata_with_openai(question_content, explanation)
    
    # 所有嘗試都失敗，使用本地提取作為最後的備用選項
    print("所有API嘗試均失敗，使用本地提取方法")
    return enhanced_fallback_extraction(question_content, explanation)

# 修改元數據提取函數以支持模型選擇
def extract_legal_metadata(question_content, explanation="", model_name=None):
    """
    提取題目的法律元數據，支持指定模型
    """
    try:
        # 如果未指定模型，使用默認模型
        if not model_name:
            model_name = GEMINI_MODELS["text"]["default"]
            
        model = genai.GenerativeModel(model_name)
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
        - 考點必須具體明確，不要過於籠統
        - 關鍵字必須是法律術語，避免使用一般性詞彙
        - 如無明確法條，可以使用相關法律原則或學說
        """.format(question=question_content, explanation=explanation)
        
        response = model.generate_content(prompt)
        result_text = response.text
        
        # 提取JSON字串
        import re
        import json
        
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            json_str = json_match.group(0)
            try:
                metadata = json.loads(json_str)
                
                # 確保結果包含預期的欄位
                if "exam_point" not in metadata or "keywords" not in metadata or "law_references" not in metadata:
                    raise ValueError("提取的元數據缺少必要欄位")
                
                # 確保關鍵字和法條參考是列表
                if not isinstance(metadata.get("keywords", []), list):
                    metadata["keywords"] = []
                if not isinstance(metadata.get("law_references", []), list):
                    metadata["law_references"] = []
                
                return metadata
            except json.JSONDecodeError:
                raise ValueError(f"無法解析Gemini返回的JSON: {json_str}")
        else:
            # 如果找不到JSON，嘗試更靈活的提取方式
            exam_point_match = re.search(r'"exam_point"\s*:\s*"([^"]+)"', result_text)
            keywords_match = re.search(r'"keywords"\s*:\s*\[(.*?)\]', result_text, re.DOTALL)
            law_refs_match = re.search(r'"law_references"\s*:\s*\[(.*?)\]', result_text, re.DOTALL)
            
            metadata = {}
            
            if exam_point_match:
                metadata["exam_point"] = exam_point_match.group(1)
            else:
                metadata["exam_point"] = ""
                
            if keywords_match:
                keywords_str = keywords_match.group(1)
                keywords = re.findall(r'"([^"]+)"', keywords_str)
                metadata["keywords"] = keywords
            else:
                metadata["keywords"] = []
                
            if law_refs_match:
                law_refs_str = law_refs_match.group(1)
                law_refs = re.findall(r'"([^"]+)"', law_refs_str)
                metadata["law_references"] = law_refs
            else:
                metadata["law_references"] = []
            
            return metadata
    except Exception as e:
        print(f"使用 {model_name} 提取法律元數據失敗: {str(e)}")
        raise

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

def process_text_with_gemini(text_blocks, multi_doc=False):
    """
    使用Gemini模型處理文本塊，提取結構化問題數據
    
    Args:
        text_blocks: 要處理的文本塊列表
        multi_doc: 是否處理為多文檔
        
    Returns:
        包含結構化問題的列表
    """
    try:
        # 嘗試導入增強的JSON處理模塊
        from process_json import clean_for_json, safe_json_dumps, extract_json_from_text
        use_enhanced_json = True
        print("使用增強的JSON處理功能")
    except ImportError:
        use_enhanced_json = False
        print("無法導入增強的JSON處理功能，使用標準處理")
        
    try:
        all_questions = []
        
        for i, block in enumerate(text_blocks):
            print(f"處理文本塊 {i+1}/{len(text_blocks)}，長度: {len(block)} 字元")
            block_questions = process_block(block, i+1, start_id=len(all_questions)+1)
            
            # 檢查返回結果是否有效
            if isinstance(block_questions, list):
                # 清理每個問題的數據
                for q in block_questions:
                    if use_enhanced_json:
                        # 使用增強的清理函數處理每個問題
                        q = clean_for_json(q)
                    else:
                        # 使用現有函數處理None值
                        if q is not None:
                            q = clean_metadata_for_json(q)
                
                all_questions.extend(block_questions)
                print(f"已從區塊 {i+1} 提取 {len(block_questions)} 道題目，總計: {len(all_questions)} 道題目")
            
        # 如果沒有找到問題，返回空列表
        if not all_questions:
            print("未能提取到任何題目，返回空列表")
            return []
            
        print(f"成功提取 {len(all_questions)} 道題目")
        
        # 確保每個問題都有所需的屬性
        result = {"questions": []}
        for q in all_questions:
            # 跳過空值
            if q is None:
                continue
                
            # 確保數據完整性
            if "id" not in q:
                q["id"] = len(result["questions"]) + 1
                
            # 確保基本屬性存在
            if "content" not in q or not q["content"]:
                q["content"] = "未能識別題目內容"
            if "options" not in q or not isinstance(q["options"], dict):
                q["options"] = {}
            if "answer" not in q:
                q["answer"] = ""
                
            result["questions"].append(q)
        
        # 額外檢查是否有語法錯誤
        try:
            # 測試序列化
            if use_enhanced_json:
                # 使用增強的JSON處理
                json_str = safe_json_dumps(result, ensure_ascii=False)
            else:
                # 使用標準JSON處理
                json_str = json.dumps(result, ensure_ascii=False)
        except Exception as json_error:
            print(f"序列化錯誤: {str(json_error)}")
            # 如果序列化失敗，嘗試更嚴格的清理
            if use_enhanced_json:
                # 使用增強的清理
                result = clean_for_json(result)
                try:
                    # 再次嘗試序列化
                    json_str = safe_json_dumps(result, ensure_ascii=False)
                except Exception as retry_error:
                    print(f"即使使用增強清理後仍然序列化失敗: {str(retry_error)}")
                    result = {"questions": [], "warning": f"序列化失敗: {str(retry_error)}"}
            else:
                # 使用常規清理
                try:
                    clean_questions = []
                    for q in result["questions"]:
                        clean_questions.append(clean_metadata_for_json(q))
                    result = {"questions": clean_questions}
                except Exception:
                    result = {"questions": [], "warning": f"序列化失敗: {str(json_error)}"}
        
        return result
        
    except Exception as e:
        error_message = f"使用Gemini API處理文字時發生錯誤: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return {"error": error_message}

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

def retrieve_questions_by_keyword(keyword, num_questions=10, expand_keywords=True):
    """
    根據關鍵字檢索題目 - 增強版
    
    主要改進：
    1. 使用雙模型架構：Gemini嵌入向量用於檢索，Gemini LLM用於關鍵詞擴展
    2. 關鍵詞擴展：根據原始關鍵詞生成多個相關法律概念
    3. 混合檢索策略：綜合語義相似度、關鍵詞匹配、考點匹配和法條匹配
    4. 相關性排序：根據多種因素對檢索結果進行權重排序
    5. 錯誤處理與備用策略
    
    Args:
        keyword: 關鍵字
        num_questions: 需要的題目數量
        expand_keywords: 是否使用LLM擴展關鍵詞
    
    Returns:
        list: 符合關鍵字的題目列表，按相關性排序
    """
    from vector_store import vector_store
    import google.generativeai as genai
    import time
    import os
    
    print(f"開始智能檢索關鍵字 '{keyword}' 相關題目")
    
    # 步驟1: 關鍵詞擴展（使用LLM生成相關法律概念）
    expanded_keywords = [keyword]
    if expand_keywords:
        try:
            # 配置Gemini
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # 生成擴展關鍵詞
                prompt = f"""
                你是一位專業的法律教育專家。請根據以下關鍵詞，生成3-5個相關的法律概念或術語，這些概念在法律考試中經常與主關鍵詞一同出現。
                請確保這些概念具有法律專業性，並與原關鍵詞有明確的關聯。

                關鍵詞: {keyword}

                請直接列出這些相關概念，每行一個，不要有編號或其他標記。
                """
                
                response = model.generate_content(prompt)
                expansion_text = response.text.strip()
                
                # 解析擴展關鍵詞
                potential_keywords = [kw.strip() for kw in expansion_text.split('\n') if kw.strip()]
                # 過濾掉太長或太短的關鍵詞
                filtered_keywords = [kw for kw in potential_keywords if 2 <= len(kw) <= 15]
                
                if filtered_keywords:
                    expanded_keywords.extend(filtered_keywords)
                    print(f"擴展關鍵詞: {filtered_keywords}")
        except Exception as e:
            print(f"關鍵詞擴展失敗: {str(e)}，將使用原始關鍵詞")
    
    # 步驟2: 多管道檢索策略
    all_results = []
    seen_question_ids = set()
    
    # 檢索方法1: 語義相似度搜索
    semantic_results = vector_store.search_similar_questions(keyword, num_questions * 2)
    for result in semantic_results:
        question_id = result["metadata"].get("question_number")
        if question_id not in seen_question_ids:
            result["score"] = 1.0 - (result.get("distance", 0) or 0)  # 轉換距離為相似度分數
            result["match_type"] = "語義相似"
            all_results.append(result)
            seen_question_ids.add(question_id)
    
    # 檢索方法2: 對每個擴展關鍵詞進行關鍵詞匹配
    for exp_keyword in expanded_keywords:
        keyword_results = vector_store.search_by_keyword(exp_keyword, num_questions)
        
        for result in keyword_results:
            question_id = result["metadata"].get("question_number")
            if question_id not in seen_question_ids:
                # 如果是原始關鍵詞，給予更高權重
                weight = 0.9 if exp_keyword == keyword else 0.7
                result["score"] = weight
                result["match_type"] = f"關鍵詞匹配:{exp_keyword}"
                all_results.append(result)
                seen_question_ids.add(question_id)
    
    # 檢索方法3: 考點匹配
    exam_point_results = vector_store.search_by_exam_point(keyword, num_questions // 2)
    for result in exam_point_results:
        question_id = result["metadata"].get("question_number")
        if question_id not in seen_question_ids:
            result["score"] = 0.85  # 考點匹配給較高分數
            result["match_type"] = "考點匹配"
            all_results.append(result)
            seen_question_ids.add(question_id)
    
    # 檢索方法4: 法條引用匹配
    # 只在關鍵詞可能是法條時使用
    if "法" in keyword or "條" in keyword:
        law_results = vector_store.search_by_law_reference(keyword, num_questions // 2)
        for result in law_results:
            question_id = result["metadata"].get("question_number")
            if question_id not in seen_question_ids:
                result["score"] = 0.8
                result["match_type"] = "法條匹配"
                all_results.append(result)
                seen_question_ids.add(question_id)
    
    # 步驟3: 根據多種因素進行權重排序
    def calculate_final_score(result):
        base_score = result.get("score", 0.5)
        metadata = result["metadata"]
        
        # 增加額外的權重因素
        bonus = 0
        
        # 1. 如果考點包含關鍵詞，加分
        exam_point = metadata.get("exam_point", "")
        if any(kw in exam_point for kw in expanded_keywords):
            bonus += 0.15
        
        # 2. 如果選項多且答案解析完整，加分
        options = metadata.get("options", {})
        if isinstance(options, str):
            try:
                import json
                options = json.loads(options)
            except:
                options = {}
        
        explanation = metadata.get("explanation", "")
        if len(options) >= 4 and len(explanation) > 30:
            bonus += 0.1
        
        # 3. 按匹配類型調整分數
        match_type = result.get("match_type", "")
        if "語義相似" in match_type:
            bonus += 0.05
        elif "關鍵詞匹配" in match_type and keyword in match_type:
            bonus += 0.1
        
        return base_score + bonus
    
    # 計算最終分數並排序
    for result in all_results:
        result["final_score"] = calculate_final_score(result)
    
    all_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # 步驟4: 格式化結果
    formatted_questions = []
    for idx, result in enumerate(all_results[:num_questions], 1):
        metadata = result["metadata"]
        content = result["content"]
        
        # 處理選項，確保格式正確
        options = metadata.get("options", {})
        if isinstance(options, str):
            try:
                import json
                options = json.loads(options)
            except:
                options = {}
        
        # 處理其他元數據字段
        keywords = metadata.get("keywords", [])
        if isinstance(keywords, str):
            try:
                import json
                keywords = json.loads(keywords)
            except:
                keywords = []
        
        law_references = metadata.get("law_references", [])
        if isinstance(law_references, str):
            try:
                import json
                law_references = json.loads(law_references)
            except:
                law_references = []
        
        question = {
            "id": idx,  # 為題目分配新的ID
            "content": content,
            "options": options,
            "answer": metadata.get("answer", ""),
            "explanation": metadata.get("explanation", ""),
            "exam_point": metadata.get("exam_point", ""),
            "keywords": keywords,
            "law_references": law_references,
            "type": metadata.get("type", "單選題"),
            "source": f"來自 {metadata.get('exam_name', '未知來源')}",
            "match_info": result.get("match_type", "關鍵詞相關")
        }
        formatted_questions.append(question)
    
    print(f"智能檢索完成，找到 {len(formatted_questions)} 道相關題目，來自 {len(all_results)} 個檢索結果")
    return formatted_questions

def format_question_for_llm(question):
    """
    將題目格式化為適合LLM處理的格式
    
    Args:
        question: 題目數據
    
    Returns:
        str: 格式化後的題目文本
    """
    formatted = f"題目：{question['content']}\n"
    
    # 添加選項
    if question['options']:
        formatted += "選項：\n"
        for key, value in question['options'].items():
            formatted += f"{key}. {value}\n"
    
    # 添加答案和解析
    formatted += f"答案：{question['answer']}\n"
    if question['explanation']:
        formatted += f"解析：{question['explanation']}\n"
    
    # 添加元數據
    if question['exam_point']:
        formatted += f"考點：{question['exam_point']}\n"
    
    if question['keywords']:
        keywords_str = "、".join(question['keywords'])
        formatted += f"關鍵詞：{keywords_str}\n"
    
    if question['law_references']:
        law_refs_str = "；".join(question['law_references'])
        formatted += f"法律依據：{law_refs_str}\n"
    
    return formatted

def generate_questions_with_llm(keyword, count=5, examples=None):
    """
    使用LLM生成新的題目
    
    Args:
        keyword: 關鍵字
        count: 需要生成的題目數量
        examples: 示例題目
    
    Returns:
        list: 生成的題目列表
    """
    import os
    import google.generativeai as genai
    import json
    import random
    
    # 檢查是否配置了API密鑰
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("警告：未找到 GOOGLE_API_KEY 環境變量，無法使用 Gemini 生成題目")
        return []
    
    # 配置 Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # 準備提示詞
    system_prompt = """你是一位專業的法律考試命題專家。請根據提供的關鍵字和參考題目，創建全新的法律選擇題。

每個題目必須包含：
1. 題目內容
2. 四個選項 (A, B, C, D)
3. 正確答案
4. 詳細的解析說明
5. 考點
6. 相關法條依據
7. 關鍵詞（3-5個）

題目要求：
- 題目難度要適中
- 內容要專業準確
- 選項要有區分度
- 解析要清晰易懂
- 必須與關鍵字主題相關
- 題目必須是原創的，不得直接複製參考題目

輸出格式：以 JSON 數組形式輸出，每個題目包含 id, content, options, answer, explanation, exam_point, keywords, law_references, type 字段。
"""

    # 構建用戶提示詞
    user_prompt = f"請根據關鍵字「{keyword}」生成 {count} 道全新的法律選擇題。"
    
    if examples and len(examples) > 0:
        user_prompt += "\n\n以下是一些參考題目格式：\n"
        for i, example in enumerate(examples, 1):
            user_prompt += f"參考題目 {i}：\n{example}\n\n"
    
    try:
        # 調用 Gemini API 生成題目
        response = model.generate_content([
            {"role": "system", "parts": [system_prompt]},
            {"role": "user", "parts": [user_prompt]}
        ])
        
        # 解析生成的內容
        generated_text = response.text
        
        # 嘗試提取JSON
        # 找出文本中的JSON數組
        import re
        json_match = re.search(r'\[\s*{.*}\s*\]', generated_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            questions = json.loads(json_str)
        else:
            # 如果沒有匹配到標準JSON格式，嘗試其他提取方法
            # 去除可能干擾 JSON 解析的 markdown 代碼塊標記
            clean_text = generated_text.replace("```json", "").replace("```", "").strip()
            # 嘗試直接解析
            questions = json.loads(clean_text)
        
        # 確保每個題目都有必要的字段
        formatted_questions = []
        for i, q in enumerate(questions, 1):
            # 為生成的題目設置原始 ID
            if "id" not in q:
                q["id"] = i
            
            # 確保每個題目都有完整的選項
            if "options" not in q or not q["options"]:
                q["options"] = {"A": "選項A", "B": "選項B", "C": "選項C", "D": "選項D"}
            
            # 確保有答案
            if "answer" not in q or not q["answer"]:
                q["answer"] = random.choice(["A", "B", "C", "D"])
            
            # 確保有題目類型
            if "type" not in q:
                q["type"] = "單選題"
            
            # 添加其他必要字段
            if "explanation" not in q:
                q["explanation"] = "該題目考察了法律知識的應用。"
            
            if "exam_point" not in q:
                q["exam_point"] = keyword
            
            if "keywords" not in q or not q["keywords"]:
                q["keywords"] = [keyword]
            
            if "law_references" not in q or not q["law_references"]:
                q["law_references"] = ["相關法條"]
            
            formatted_questions.append(q)
        
        print(f"成功使用 Gemini 生成了 {len(formatted_questions)} 道新題目")
        return formatted_questions
    
    except Exception as e:
        print(f"使用 Gemini 生成題目時發生錯誤：{str(e)}")
        import traceback
        traceback.print_exc()
        return []

# 修改嵌入向量函數實現，加入智能降級機制
class GeminiEmbeddingFunction:
    """Gemini API嵌入函數類，提供向量化能力"""
    
    def __init__(self, model_name=None, api_key=None, batch_size=None):
        """初始化Gemini嵌入函數
        
        Args:
            model_name: 嵌入模型名稱，不指定則使用默認模型
            api_key: Google API密鑰，不指定則從環境變量獲取
            batch_size: 批處理大小，不指定則使用默認值
        """
        # 獲取API密鑰
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("未設置Google API密鑰，請設置GOOGLE_API_KEY環境變量")
        
        # 確保有效模型名稱
        default_model = GEMINI_MODELS["embedding"]["default"]
        self.model_name = model_name or default_model
        
        # 確保模型名稱格式正確，添加"models/"前綴
        if not self.model_name.startswith("models/") and not self.model_name.startswith("tunedModels/"):
            self.model_name = f"models/{self.model_name}"
            
        # 設置批處理大小
        self.batch_size = batch_size or GEMINI_MODELS["embedding"]["batch_size"]
        
        # 初始化Google Generative AI客戶端
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.genai = genai
        
        # 測試API連接和模型可用性
        try:
            # 測試嵌入功能
            embedding = self._get_embeddings_with_gemini(["測試連接"])
            if not embedding or len(embedding) == 0 or len(embedding[0]) == 0:
                raise ValueError("嵌入模型返回空結果")
            print(f"Gemini嵌入模型 '{self.model_name}' 初始化成功，批次大小：{self.batch_size}")
        except Exception as e:
            print(f"無法初始化Gemini嵌入模型: {str(e)}")
            raise
        
        # 追蹤API使用情況
        self.api_calls = 0
        self.fallback_calls = 0
        
        print(f"初始化 GeminiEmbeddingFunction，使用模型 {self.model_name}，批次大小 {self.batch_size}")
    
    def __call__(self, texts):
        """執行嵌入處理，支持單一文本或文本列表"""
        if not texts:
            return []
        
        # 規範化輸入為列表
        if isinstance(texts, str):
            texts = [texts]
        
        # 過濾空文本
        texts = [text for text in texts if text and isinstance(text, str)]
        if not texts:
            return []
        
        # 使用批次處理
        return self._process_in_batches(texts)
    
    def _process_in_batches(self, texts):
        """批次處理文本嵌入"""
        import time
        import numpy as np
        
        # 分批處理
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        all_embeddings = []
        for i, batch in enumerate(batches):
            print(f"處理第 {i+1}/{len(batches)} 批嵌入...")
            
            # 嘗試使用Gemini API
            try:
                batch_embeddings = self._get_embeddings_with_gemini(batch)
                all_embeddings.extend(batch_embeddings)
                self.api_calls += 1
            except Exception as e:
                # API調用失敗時使用本地模型
                print(f"Gemini嵌入API調用失敗: {str(e)}")
                if self.local_model:
                    print("使用本地嵌入模型作為備用")
                    batch_embeddings = self._get_embeddings_with_local_model(batch)
                    all_embeddings.extend(batch_embeddings)
                    self.fallback_calls += 1
                else:
                    # 沒有本地模型時，使用全零向量作為佔位符
                    print("無本地模型，使用零向量佔位")
                    for _ in batch:
                        all_embeddings.append(np.zeros(768))  # 標準嵌入向量大小
            
            # 除最後一批外，添加短暫延遲以避免API限制
            if i < len(batches) - 1:
                time.sleep(1.0)
        
        return all_embeddings
    
    def _get_embeddings_with_gemini(self, texts):
        """使用Google Gemini API獲取嵌入向量"""
        import numpy as np
        import google.generativeai as genai
        
        try:
            # 確保模型名稱格式正確
            model_name = self.model_name
            if not model_name.startswith("models/") and not model_name.startswith("tunedModels/"):
                # 添加適當的前綴
                model_name = f"models/{model_name}"
            
            print(f"使用模型 {model_name} 生成嵌入向量")
            
            # 設置API密鑰
            genai.configure(api_key=self.api_key)
            
            # 使用Gemini嵌入API
            embeddings = []
            for text in texts:
                try:
                    # 獲取嵌入向量
                    embedding = genai.get_embedding(
                        model=model_name,
                        content=text
                    )
                    # 轉換為NumPy數組
                    embeddings.append(np.array(embedding["values"]))
                except Exception as e:
                    print(f"處理文本嵌入時出錯: {str(e)}")
                    # 返回零向量作為替代
                    embeddings.append(np.zeros(768))  # 使用常見的嵌入維度
            
            return embeddings
        
        except Exception as e:
            print(f"嵌入API錯誤: {str(e)}")
            # 如果API出錯，使用本地模型作為備用
            print("嘗試使用本地嵌入模型作為備用")
            return self._get_embeddings_with_local_model(texts)
    
    def _get_embeddings_with_local_model(self, texts):
        """使用本地SentenceTransformer模型獲取嵌入"""
        import numpy as np
        
        # 確保本地模型已加載
        if not self.local_model:
            raise ValueError("本地嵌入模型未初始化")
        
        # 本地模型支持批量處理
        return self.local_model.encode(texts)

# 修改向量庫使用方式
def get_embedding_function(model_name=None, batch_size=None):
    """
    獲取嵌入函數，支持配置和自動降級
    """
    try:
        # 嘗試使用Gemini嵌入
        embedding_function = GeminiEmbeddingFunction(
            model_name=model_name or GEMINI_MODELS["embedding"]["default"],
            batch_size=batch_size or GEMINI_MODELS["embedding"]["batch_size"]
        )
        print(f"使用Gemini嵌入模型: {embedding_function.model_name}")
        return embedding_function
    except Exception as e:
        print(f"無法初始化Gemini嵌入，使用SentenceTransformer作為後備: {str(e)}")
        # 使用SentenceTransformer作為後備
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode

# 在保存JSON前添加這個函數
def clean_metadata_for_json(obj):
    """深度清理元數據，確保可以被JSON序列化
    處理：
    - None值 -> 空字符串
    - NumPy數組 -> Python列表
    - NumPy數值類型 -> Python原生類型
    - 嵌套字典和列表
    - 其他不可序列化類型 -> 字符串表示
    """
    import numpy as np
    import json
    
    # 處理None值
    if obj is None:
        return ""
        
    # 專門處理可能導致"The truth value of an array"錯誤的對象
    try:
        # 檢查是否有NumPy庫
        if 'numpy' in sys.modules or np:
            # 處理NumPy數組
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # 轉換為Python列表
                
            # 處理NumPy數值類型
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()  # 轉換為Python原生類型
                
            # 處理布爾數組的條件判斷
            if hasattr(obj, 'any') and hasattr(obj, 'all'):
                try:
                    # 嘗試判斷是否為數組
                    if hasattr(obj, 'shape') or hasattr(obj, '__array__'):
                        return np.asarray(obj).tolist()
                except:
                    pass
    except (NameError, ImportError, AttributeError) as e:
        # 處理可能的導入或屬性錯誤
        print(f"處理NumPy類型時出錯: {str(e)}")
    except Exception as e:
        print(f"未預期的錯誤: {str(e)}")
    
    # 處理字典
    if isinstance(obj, dict):
        # 創建一個新字典，跳過None鍵，清理所有值
        return {
            k: clean_metadata_for_json(v) 
            for k, v in obj.items() 
            if k is not None  # 跳過None鍵
        }
    
    # 處理列表或元組
    if isinstance(obj, (list, tuple)):
        return [clean_metadata_for_json(item) for item in obj]
    
    # 處理集合類型
    if isinstance(obj, set):
        return [clean_metadata_for_json(item) for item in obj]
    
    # 嘗試確保其他類型可以被JSON序列化
    try:
        # 測試是否為可序列化類型
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        # 對於無法序列化的類型，先嘗試常見轉換
        try:
            # 嘗試將對象轉換為字典
            if hasattr(obj, '__dict__'):
                return clean_metadata_for_json(obj.__dict__)
            # 嘗試使用通用方法轉換
            elif hasattr(obj, 'tolist'):
                return clean_metadata_for_json(obj.tolist())
            elif hasattr(obj, 'to_dict'):
                return clean_metadata_for_json(obj.to_dict())
            # 檢查是否可以轉換為列表
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
                return [clean_metadata_for_json(item) for item in obj]
            else:
                # 最後嘗試字符串轉換
                return str(obj)
        except Exception as e:
            print(f"處理不可序列化對象時出錯: {str(e)}")
            # 如果所有嘗試都失敗，返回空字符串
            return ""

# 在use_gemini.py約1436行，return metadata前添加
def clean_none_values(obj):
    """清理元數據中的None值"""
    if isinstance(obj, dict):
        return {k: clean_none_values(v) if v is not None else "" for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_none_values(i) if i is not None else "" for i in obj]
    else:
        return obj if obj is not None else ""

# 在GeminiEmbeddingFunction類之後添加
class OpenAIEmbeddingFunction:
    """OpenAI嵌入模型作為備選方案"""
    
    def __init__(self, model_name="text-embedding-3-small", api_key=None, batch_size=None):
        """初始化OpenAI嵌入模型
        
        Args:
            model_name: OpenAI嵌入模型名稱
            api_key: OpenAI API密鑰，不指定則從環境變量獲取
            batch_size: 批處理大小
        """
        # 獲取API密鑰
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("未設置OpenAI API密鑰，請設置OPENAI_API_KEY環境變量")
        
        self.model_name = model_name
        self.batch_size = batch_size or 20  # OpenAI有更高的批處理能力
        
        # 初始化客戶端
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            # 測試連接
            _ = self.__call__(["測試連接"])
            print(f"OpenAI嵌入模型 '{self.model_name}' 初始化成功，批次大小：{self.batch_size}")
        except Exception as e:
            print(f"OpenAI嵌入模型初始化失敗: {str(e)}")
            raise
    
    def __call__(self, texts):
        """支持與GeminiEmbeddingFunction相同的調用接口"""
        if isinstance(texts, str):
            texts = [texts]
        
        # 使用分批處理以處理大量文本
        if len(texts) > self.batch_size:
            return self._process_in_batches(texts)
        else:
            return self._get_embeddings(texts)
    
    def _process_in_batches(self, texts):
        """分批處理文本"""
        all_embeddings = []
        
        # 按批次處理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_embeddings = self._get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _get_embeddings(self, texts):
        """使用OpenAI API獲取嵌入向量"""
        import numpy as np
        
        try:
            # 取得嵌入向量
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                dimensions=1536  # OpenAI默認維度
            )
            
            # 提取嵌入向量
            embeddings = [np.array(item.embedding) for item in response.data]
            return embeddings
        
        except Exception as e:
            print(f"OpenAI嵌入API錯誤: {str(e)}")
            raise

# 修改向量庫使用方式
def get_embedding_function(model_name=None, batch_size=None):
    """
    獲取嵌入函數，支持配置和多級自動降級
    順序: Gemini -> OpenAI -> SentenceTransformer
    """
    # 1. 嘗試使用Gemini嵌入
    try:
        embedding_function = GeminiEmbeddingFunction(
            model_name=model_name or GEMINI_MODELS["embedding"]["default"],
            batch_size=batch_size or GEMINI_MODELS["embedding"]["batch_size"]
        )
        print(f"使用Gemini嵌入模型: {embedding_function.model_name}")
        return embedding_function
    except Exception as e:
        print(f"無法初始化Gemini嵌入: {str(e)}")
        
        # 2. 嘗試使用OpenAI作為第一備選
        try:
            print("嘗試使用OpenAI嵌入作為備選...")
            embedding_function = OpenAIEmbeddingFunction(
                batch_size=batch_size or 20
            )
            return embedding_function
        except Exception as e:
            print(f"無法初始化OpenAI嵌入: {str(e)}")
            
            # 3. 使用SentenceTransformer作為最後備選
            print("使用SentenceTransformer作為最終備選")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode

# 添加提取法律元數據的函數
def extract_metadata_single_doc(text, model=None):
    """從單個文檔提取法律元數據
    
    Args:
        text: 文本內容
        model: Gemini模型名稱
        
    Returns:
        包含元數據的字典
    """
    try:
        model_name = model or GEMINI_MODELS["text"]["default"]
        
        prompt = f"""
        請從以下法律文本中提取關鍵元數據。
        分析文本結構，提取以下信息（如有）：
        1. 文件類型（如法律、條例、合同、判決書等）
        2. 發布/生效日期
        3. 標題/名稱
        4. 來源機構/組織
        5. 主要主題
        6. 關鍵條款或章節摘要
        
        以JSON格式返回結果，格式如下：
        {{
            "文件類型": "...",
            "發布日期": "...",
            "生效日期": "...",
            "標題": "...",
            "來源": "...",
            "主題": ["...", "..."],
            "關鍵條款": [{{"條款編號": "...", "內容摘要": "..."}}],
            "其他特殊字段": "..."
        }}
        
        如果某個字段不存在，請使用空字符串或空數組。
        文本內容：
        {text[:10000]}
        """
        
        # 調用Gemini模型
        response = process_with_gemini(prompt, model_name, GEMINI_MODELS["text"]["batch_size"])
        
        # 提取返回的JSON數據
        metadata = extract_json_from_text(response)
        
        # 如果沒有成功解析JSON，使用最小結構
        if not metadata:
            metadata = {
                "文件類型": "",
                "標題": "",
                "內容預覽": text[:100] + "..." if len(text) > 100 else text,
                "處理狀態": "僅部分處理"
            }
        
        return metadata
    
    except Exception as e:
        print(f"提取單文檔元數據失敗: {str(e)}")
        return {
            "處理狀態": "失敗",
            "錯誤": str(e),
            "內容預覽": text[:100] + "..." if len(text) > 100 else text
        }

def extract_document_structure(text, model=None):
    """從多文檔文本中提取文檔結構和元數據
    
    Args:
        text: 文本內容
        model: Gemini模型名稱
        
    Returns:
        包含文檔結構和元數據的字典
    """
    try:
        model_name = model or GEMINI_MODELS["text"]["default"]
        
        prompt = f"""
        請分析以下文本，可能包含多個不同的法律文檔或章節。
        識別每個獨立文檔或主要章節的邊界，並提取各自的元數據。
        
        對每個識別的文檔/章節，提取：
        1. 文檔類型
        2. 標題/名稱
        3. 開始位置（大致字符位置或關鍵標識詞）
        4. 結束位置（大致字符位置或關鍵標識詞）
        5. 核心內容摘要（50字以內）
        
        以JSON格式返回結果：
        {{
            "文檔數量": 數量,
            "文檔列表": [
                {{
                    "序號": 1,
                    "類型": "...",
                    "標題": "...",
                    "開始標識": "...",
                    "結束標識": "...",
                    "摘要": "..."
                }},
                ...
            ]
        }}
        
        文本內容：
        {text[:15000]}
        """
        
        # 調用Gemini模型
        response = process_with_gemini(prompt, model_name, GEMINI_MODELS["text"]["batch_size"])
        
        # 提取返回的JSON數據
        structure = extract_json_from_text(response)
        
        # 如果沒有成功解析JSON，使用最小結構
        if not structure:
            structure = {
                "文檔數量": 1,
                "文檔列表": [
                    {
                        "序號": 1,
                        "類型": "未知",
                        "標題": "未能識別",
                        "摘要": text[:100] + "..." if len(text) > 100 else text
                    }
                ]
            }
        
        return structure
    
    except Exception as e:
        print(f"提取文檔結構失敗: {str(e)}")
        return {
            "文檔數量": 0,
            "文檔列表": [],
            "處理狀態": "失敗",
            "錯誤": str(e)
        }

def extract_json_from_text(text):
    """從文本中提取JSON內容並修復常見格式問題
    
    處理各種常見的JSON格式錯誤，包括：
    - 單引號而非雙引號
    - 屬性名沒有使用引號
    - 缺少逗號或括號不匹配
    - 其他格式問題
    """
    import json
    import re
    
    # 如果輸入為空，直接返回空字典
    if not text or not isinstance(text, str):
        return {}
    
    # 移除Markdown代碼塊符號
    text = re.sub(r'```(?:json)?\s*|\s*```', '', text)
    
    try:
        # 尋找JSON對象的起始和結束位置
        json_pattern = r'(\{[\s\S]*\})'
        match = re.search(json_pattern, text)
        
        if match:
            json_str = match.group(1)
            
            # 嘗試直接解析
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON格式錯誤，嘗試修復: {str(e)}")
                original_error = str(e)
                error_position_match = re.search(r'line (\d+) column (\d+)', original_error)
                
                # 如果錯誤信息中包含"Expecting property name enclosed in double quotes"
                if "Expecting property name enclosed in double quotes" in original_error and error_position_match:
                    try:
                        line_num = int(error_position_match.group(1))
                        col_num = int(error_position_match.group(2))
                        
                        # 分割文本為行
                        lines = json_str.split('\n')
                        if line_num <= len(lines):
                            # 獲取問題行
                            problem_line = lines[line_num - 1]
                            
                            # 在問題位置添加引號
                            if col_num <= len(problem_line):
                                # 找到未加引號的屬性名
                                property_match = re.search(r'(\w+)\s*:', problem_line[col_num-1:])
                                if property_match:
                                    property_name = property_match.group(1)
                                    # 替換未加引號的屬性名為帶引號的形式
                                    fixed_line = problem_line[:col_num-1] + problem_line[col_num-1:].replace(
                                        f"{property_name}:", f'"{property_name}":', 1
                                    )
                                    lines[line_num - 1] = fixed_line
                                    # 重新組合JSON字符串
                                    json_str = '\n'.join(lines)
                                    print(f"已修復行 {line_num} 的屬性名引號問題")
                    except Exception as fix_error:
                        print(f"嘗試修復特定位置的JSON錯誤時失敗: {str(fix_error)}")
                
                # 進行更全面的修復
                # 1. 替換單引號為雙引號（但避免替換已在雙引號內的單引號）
                in_quotes = False
                fixed_str = []
                for i, char in enumerate(json_str):
                    if char == '"' and (i == 0 or json_str[i-1] != '\\'):
                        in_quotes = not in_quotes
                    elif char == "'" and not in_quotes:
                        fixed_str.append('"')
                    else:
                        fixed_str.append(char)
                json_str = ''.join(fixed_str)
                
                # 2. 確保屬性名使用雙引號 (處理沒有引號的屬性名)
                # 更強大的正則表達式，捕獲更多種情況的無引號屬性
                json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', json_str)
                # 特別處理可能出現在行首的無引號屬性名
                json_str = re.sub(r'(^|\n)(\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1\2"\3"\4', json_str)
                
                # 3. 清理不必要的空格和換行，但保留引號內的內容
                # 先處理引號外的空白
                in_quotes = False
                clean_str = []
                for char in json_str:
                    if char == '"' and (not clean_str or clean_str[-1] != '\\'):
                        in_quotes = not in_quotes
                    if not in_quotes and char.isspace():
                        if not clean_str or clean_str[-1] != ' ':
                            clean_str.append(' ')
                    else:
                        clean_str.append(char)
                json_str = ''.join(clean_str)
                
                # 4. 修復常見的逗號問題
                # 修復多餘的逗號
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                # 修復缺少的逗號
                json_str = re.sub(r'}\s*{', '},{', json_str)
                json_str = re.sub(r']\s*{', '],{', json_str)
                json_str = re.sub(r'}\s*\[', '},\[', json_str)
                json_str = re.sub(r']\s*\[', '],\[', json_str)
                
                # 5. 修復不完整的JSON (如果最後缺少大括號)
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                if open_braces > close_braces:
                    json_str += "}" * (open_braces - close_braces)
                
                # 6. 修復列表中多餘的逗號
                json_str = re.sub(r',(\s*,)+', ',', json_str)
                
                # 最終嘗試解析修復後的JSON
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e2:
                    print(f"修復後的JSON仍然無效: {str(e2)}, 修復後內容: {json_str}")
                    
                    # 最後嘗試從文本中提取關鍵值對
                    result = {}
                    key_value_pattern = r'"([^"]+)"\s*:\s*(?:"([^"]*)"|\[([^\]]*)\]|(\d+)|true|false|null)'
                    for match in re.finditer(key_value_pattern, json_str):
                        key = match.group(1)
                        value = match.group(2) or match.group(3) or match.group(4)
                        if value is not None:
                            if match.group(3):  # 是列表
                                items = re.findall(r'"([^"]+)"', value)
                                result[key] = items
                            elif match.group(4):  # 是數字
                                try:
                                    result[key] = int(value)
                                except:
                                    try:
                                        result[key] = float(value)
                                    except:
                                        result[key] = value
                            else:
                                result[key] = value
                    
                    return result or {"error": "解析JSON失敗", "text": text[:100]}
        
        # 如果沒有找到JSON對象，嘗試提取單個屬性
        result = {}
        key_value_pattern = r'"([^"]+)"\s*:\s*(?:"([^"]*)"|\[([^\]]*)\]|(\d+)|true|false|null)'
        for match in re.finditer(key_value_pattern, text):
            key = match.group(1)
            value = match.group(2) or match.group(3) or match.group(4)
            if value is not None:
                if match.group(3):  # 是列表
                    items = re.findall(r'"([^"]+)"', value)
                    result[key] = items
                elif match.group(4):  # 是數字
                    try:
                        result[key] = int(value)
                    except:
                        try:
                            result[key] = float(value)
                        except:
                            result[key] = value
                else:
                    result[key] = value
        
        return result or {"error": "無法找到有效的JSON結構", "text": text[:100]}
    
    except Exception as e:
        print(f"解析文本時發生未預期錯誤: {str(e)}")
        return {"error": f"解析錯誤: {str(e)}", "text": text[:100] if text else ""}

# 在文件的適當位置添加process_with_openai函數
def process_with_openai(prompt, model_name="gpt-3.5-turbo", temperature=0.3, max_retries=3, retry_delay=2, max_tokens=800):
    """使用OpenAI處理提示詞，包含完整的重試和錯誤處理
    
    Args:
        prompt: 提示詞
        model_name: OpenAI模型名稱
        temperature: 溫度參數 
        max_retries: 最大重試次數
        retry_delay: 重試延遲（秒）
        max_tokens: 最大生成令牌數
        
    Returns:
        模型回應文本
    """
    try:
        import openai
        from openai import OpenAI
        
        # 獲取API密鑰
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("未設置OpenAI API密鑰，請設置OPENAI_API_KEY環境變量")
        
        # 初始化客戶端
        client = OpenAI(api_key=api_key)
        
        for attempt in range(max_retries):
            try:
                # 調用API
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "你是一個專業的法律文本分析助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # 提取響應文本
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"OpenAI API錯誤 (嘗試 {attempt+1}/{max_retries}): {str(e)}")
                
                if attempt == max_retries - 1:
                    raise
                    
                # 指數退避等待
                time.sleep(retry_delay)
                retry_delay *= 2
        
        return "OpenAI處理失敗"
        
    except ImportError:
        raise ImportError("未安裝OpenAI模塊，請執行 pip install openai")



if __name__ == "__main__":
    # 測試用範例
    with open("sample.pdf", "rb") as f:
        pdf_bytes = f.read()
    result = process_pdf_with_gemini(pdf_bytes, "sample.pdf")
    # 在序列化前調用清理函數
    cleaned_questions = clean_metadata_for_json(result["questions"])
    json_data = json.dumps(cleaned_questions, ensure_ascii=False, indent=2)
    print(json_data)